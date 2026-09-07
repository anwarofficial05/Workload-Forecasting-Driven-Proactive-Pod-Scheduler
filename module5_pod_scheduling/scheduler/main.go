package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"
)

const (
	SchedulerName = "proactive-scheduler"
	DefaultApiUrl = "http://localhost:8000/nodes/ranking"
)

type RankedNode struct {
	Name           string  `json:"name"`
	FinalScore     float64 `json:"final_score"`
	Rank           int     `json:"rank"`
	CPUHeadroom    float64 `json:"cpu_headroom"`
	MemoryHeadroom float64 `json:"memory_headroom"`
	LatencyScore   float64 `json:"latency_score"`
	BalanceScore   float64 `json:"balance_score"`
}

type RankingResponse struct {
	Timestamp    float64      `json:"timestamp"`
	Nodes        []RankedNode `json:"nodes"`
	SelectedNode string       `json:"selected_node"`
}

type ProactiveScheduler struct {
	clientset *kubernetes.Clientset
	rankingUrl string
}

func NewProactiveScheduler(clientset *kubernetes.Clientset, rankingUrl string) *ProactiveScheduler {
	return &ProactiveScheduler{
		clientset:  clientset,
		rankingUrl: rankingUrl,
	}
}

func (s *ProactiveScheduler) fetchBestNode(ctx context.Context) (string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", s.rankingUrl, nil)
	if err != nil {
		return "", err
	}

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to call node ranking API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ranking API returned error %d: %s", resp.StatusCode, string(body))
	}

	var rankingResp RankingResponse
	if err := json.NewDecoder(resp.Body).Decode(&rankingResp); err != nil {
		return "", fmt.Errorf("failed to decode ranking JSON: %w", err)
	}

	if len(rankingResp.Nodes) == 0 {
		return "", fmt.Errorf("ranking API returned no eligible nodes")
	}

	// First node is highest ranked
	bestNode := rankingResp.Nodes[0].Name
	return bestNode, nil
}

func (s *ProactiveScheduler) bindPod(ctx context.Context, pod *corev1.Pod, nodeName string) error {
	binding := &corev1.Binding{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: pod.Namespace,
			Name:      pod.Name,
			UID:       pod.UID,
		},
		Target: corev1.ObjectReference{
			Kind:       "Node",
			APIVersion: "v1",
			Name:       nodeName,
		},
	}

	err := s.clientset.CoreV1().Pods(pod.Namespace).Bind(ctx, binding, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("failed to execute Kubernetes binding: %w", err)
	}

	// Mandatory logging format per project specification
	fmt.Printf("[SCHEDULER] Selected Node=%s\n", nodeName)
	fmt.Printf("[POD] %s scheduled on %s\n", pod.Name, nodeName)
	klog.Infof("[SCHEDULER] Selected Node=%s", nodeName)
	klog.Infof("[POD] %s scheduled on %s", pod.Name, nodeName)
	return nil
}

func (s *ProactiveScheduler) Run(ctx context.Context) {
	klog.Infof("Starting Proactive Kubernetes Pod Scheduler loop (schedulerName: %s)...", SchedulerName)
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			klog.Info("Scheduler shutting down")
			return
		case <-ticker.C:
			s.schedulePendingPods(ctx)
		}
	}
}

func (s *ProactiveScheduler) schedulePendingPods(ctx context.Context) {
	// List pods across all namespaces
	pods, err := s.clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{
		FieldSelector: "spec.nodeName=",
	})
	if err != nil {
		klog.Warningf("Failed to list pending pods: %v", err)
		return
	}

	for _, pod := range pods.Items {
		// Only handle pods configured with our custom scheduler
		if pod.Spec.SchedulerName != SchedulerName {
			continue
		}

		klog.Infof("Found pending pod %s/%s destined for proactive scheduling", pod.Namespace, pod.Name)

		bestNode, err := s.fetchBestNode(ctx)
		if err != nil {
			klog.Errorf("Node ranking evaluation failed for pod %s: %v", pod.Name, err)
			continue
		}

		err = s.bindPod(ctx, &pod, bestNode)
		if err != nil {
			klog.Errorf("Failed to bind pod %s to node %s: %v", pod.Name, bestNode, err)
		}
	}
}

func main() {
	var kubeconfig string
	var rankingUrl string

	flag.StringVar(&rankingUrl, "ranking-url", DefaultApiUrl, "URL of the Forecast & Node Ranking API")
	if home := os.Getenv("HOME"); home != "" {
		flag.StringVar(&kubeconfig, "kubeconfig", filepath.Join(home, ".kube", "config"), "Path to kubeconfig file")
	} else {
		flag.StringVar(&kubeconfig, "kubeconfig", "", "Path to kubeconfig file")
	}
	flag.Parse()

	// 1. Try In-Cluster Config (for Pod deployment)
	config, err := rest.InClusterConfig()
	if err != nil {
		// 2. Fallback to Kubeconfig (for local testing outside cluster)
		config, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
		if err != nil {
			klog.Fatalf("Error building Kubernetes client configuration: %v", err)
		}
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		klog.Fatalf("Error building Kubernetes clientset: %v", err)
	}

	scheduler := NewProactiveScheduler(clientset, rankingUrl)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	scheduler.Run(ctx)
}
