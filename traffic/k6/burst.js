import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 20 },
    { duration: '15s', target: 180 }, // Burst 1
    { duration: '30s', target: 20 },
    { duration: '15s', target: 220 }, // Burst 2
    { duration: '30s', target: 20 },
  ],
};

const BASE_URL = __ENV.TARGET_URL || 'http://localhost:8080';

export default function () {
  const res = http.get(`${BASE_URL}/?cpu_ms=40&mem_mb=10`);
  check(res, { 'status is 200': (r) => r.status === 200 });
  sleep(0.2);
}
