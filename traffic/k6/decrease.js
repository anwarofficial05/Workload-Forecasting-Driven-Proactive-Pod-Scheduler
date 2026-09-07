import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 150 }, // Initial high load
    { duration: '1m', target: 80 },  // Step down
    { duration: '1m', target: 30 },  // Low load
    { duration: '30s', target: 0 },
  ],
};

const BASE_URL = __ENV.TARGET_URL || 'http://localhost:8080';

export default function () {
  const res = http.get(`${BASE_URL}/?cpu_ms=30&mem_mb=8`);
  check(res, { 'status is 200': (r) => r.status === 200 });
  sleep(0.4);
}
