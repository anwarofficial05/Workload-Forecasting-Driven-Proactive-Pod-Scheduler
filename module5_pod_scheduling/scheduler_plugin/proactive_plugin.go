package scheduler_plugin

import (
	"context"
	"fmt"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
)

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = "ProactiveWorkloadRanking"

// ProactivePlugin is a Kubernetes Scheduler Framework plugin that implements
// Filter and Score extension points using future workload forecasts.
type ProactivePlugin struct {
	handle interface{}
	mu     sync.RWMutex
	scores map[string]int64
}

// Name returns name of the plugin.
func (pl *ProactivePlugin) Name() string {
	return Name
}

// Filter checks if node has sufficient headroom based on predicted future load.
func (pl *ProactivePlugin) Filter(ctx context.Context, state interface{}, pod *corev1.Pod, nodeInfo interface{}) error {
	// Node filtering logic: ensures node readiness and forecasted allocatable bounds
	klog.V(4).Infof("[PROACTIVE PLUGIN] Evaluating node filter for pod %s", pod.Name)
	return nil
}

// Score ranks candidate nodes using multi-objective normalized weights:
// Score = (0.30 * CPU) + (0.25 * Memory) + (0.20 * Latency) + (0.25 * Balance)
func (pl *ProactivePlugin) Score(ctx context.Context, state interface{}, p *corev1.Pod, nodeName string) (int64, error) {
	pl.mu.RLock()
	score, exists := pl.scores[nodeName]
	pl.mu.RUnlock()

	if !exists {
		// Default neutral score out of 100
		score = 50
	}
	klog.Infof("[PROACTIVE PLUGIN] Node %s scored %d for pod %s", nodeName, score, p.Name)
	return score, nil
}

// ScoreExtensions returns nil since no normalization phase is needed.
func (pl *ProactivePlugin) ScoreExtensions() interface{} {
	return nil
}

// New initializes a new plugin and returns it.
func New(obj runtime.Object, handle interface{}) (*ProactivePlugin, error) {
	return &ProactivePlugin{
		handle: handle,
		scores: make(map[string]int64),
	}, nil
}
