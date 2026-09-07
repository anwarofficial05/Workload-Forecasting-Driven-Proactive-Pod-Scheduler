import http from 'k6/http';
import { check, sleep } from 'k6';

// Most Critical Experiment:
// NORMAL TRAFFIC -> SUDDEN SPIKE -> REACTIVE VS PROACTIVE COMPARISON
export const options = {
  stages: [
    { duration: '1m', target: 30 },  // Normal baseline traffic
    { duration: '10s', target: 250 }, // SUDDEN SPIKE: 30 -> 250 VUs in 10 seconds
    { duration: '2m', target: 250 },  // Peak traffic pressure
    { duration: '30s', target: 30 },  // Traffic subsides
    { duration: '1m', target: 30 },   // Normal post-spike baseline
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // Violations occur if P95 > 500ms
  },
};

const BASE_URL = __ENV.TARGET_URL || 'http://localhost:8080';

export default function () {
  const res = http.get(`${BASE_URL}/?cpu_ms=50&mem_mb=15`);
  check(res, {
    'status is 200': (r) => r.status === 200,
    'latency within SLO': (r) => r.timings.duration < 500,
  });
  sleep(0.2);
}
