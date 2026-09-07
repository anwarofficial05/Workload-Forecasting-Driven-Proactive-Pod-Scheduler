# Stage 1: Build the Go binary
FROM golang:1.21-alpine AS builder

WORKDIR /workspace

COPY go.mod ./
# Download dependencies if network available or vendor
RUN go mod download || true

COPY main.go ./
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -a -o proactive-scheduler main.go

# Stage 2: Minimal runtime image
FROM alpine:3.19

RUN apk --no-cache add ca-certificates curl

WORKDIR /root/
COPY --from=builder /workspace/proactive-scheduler /usr/local/bin/proactive-scheduler

USER 65532:65532

ENTRYPOINT ["/usr/local/bin/proactive-scheduler"]
