# DSPy Multi-Agent API Testing Guide

This document provides comprehensive testing examples for the DSPy Multi-Agent FastAPI server.

## Prerequisites

1. **Start the FastAPI server:**

   ```bash
   python start_api.py
   # or
   uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Verify server is running:**

   ```bash
   curl -s http://localhost:8000/health | jq '.'
   ```

## API Base URL

- **Local Development**: `http://localhost:8000`
- **Production**: Replace with your production URL

## Available Endpoints

### 1. Health Check

**GET** `/health`

Check if the API server is running and healthy.

```bash
# Basic health check
curl -X GET "http://localhost:8000/health"
```

```bash
# Pretty formatted with jq
curl -s -X GET "http://localhost:8000/health" | jq '.'
```

```bash
# With response headers
curl -i -X GET "http://localhost:8000/health"
```

```bash
# Check response time
curl -w "Response time: %{time_total}s\n" -s -X GET "http://localhost:8000/health" | jq '.'
```

```bash
# Call the demo endpoint and format the JSON response
curl -s -X POST http://localhost:8000/agent/demo \
     -H "Content-Type: application/json" | \
     jq -r '
     "=============================================================",
     "TOPIC: " + .topic,
     "=============================================================",
     "",
     "SUMMARY:",
     .summary,
     "",
     "=============================================================", 
     "FULL ARTICLE:",
     .article,
     "",
     "=============================================================",
     "SOURCES:",
     (.sources | to_entries[] | "\(.key + 1). \(.value)"),
     "",
     "=============================================================",
     "📊 PROCESSING INFO:",
     "Processing Time: \(.processing_time) seconds",
     "Agent ID: \(.agent_id)",
     "Status: \(.status)",
     "✅ Demo completed successfully!"
     '
```

**Expected Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-06-23T14:35:10.956221",
  "version": "1.0.0"
}
```

---

### 2. Agent System Status

**GET** `/agents/status`

Get the current status of the multi-agent system.

```bash
# Basic status check
curl -X GET "http://localhost:8000/agents/status"

# Pretty formatted
curl -s -X GET "http://localhost:8000/agents/status" | jq '.'

# Check if all systems are ready
curl -s -X GET "http://localhost:8000/agents/status" | jq '.coordinator_ready and .model_configured'
```

**Expected Response:**

```json
{
  "coordinator_ready": true,
  "model_configured": true,
  "async_workers": 4
}
```

---

### 3. Demo Request (Built-in)

**POST** `/agent/demo`

Run the built-in demo request (same as `src/main.py`). No request body required.

```bash
# Basic demo request
curl -X POST "http://localhost:8000/agent/demo"

# Pretty formatted output
curl -s -X POST "http://localhost:8000/agent/demo" | jq '.'

# Save response to file
curl -s -X POST "http://localhost:8000/agent/demo" | jq '.' > demo_response.json

# Show only the article content
curl -s -X POST "http://localhost:8000/agent/demo" | jq -r '.article'

# Show processing time
curl -s -X POST "http://localhost:8000/agent/demo" | jq '.processing_time'

# Show formatted summary
curl -s -X POST "http://localhost:8000/agent/demo" | jq -r '"Summary: " + .summary'

# Extract and list sources
curl -s -X POST "http://localhost:8000/agent/demo" | jq -r '.sources[]' | nl

# With timing and headers
curl -w "Total time: %{time_total}s\n" -i -X POST "http://localhost:8000/agent/demo"
```

**Demo Topic:** "Write a comprehensive article about the impact of artificial intelligence on modern healthcare"

**Expected Response Structure:**

```json
{
  "status": "success",
  "topic": "Write a comprehensive article about the impact of artificial intelligence on modern healthcare",
  "article": "Full article content...",
  "summary": "Article summary...",
  "sources": ["Source 1", "Source 2", "..."],
  "key_points": ["Point 1", "Point 2", "..."],
  "processing_time": 47.2,
  "agent_id": "uuid-string"
}
```

---

### 4. Custom Process Request

**POST** `/agent/process`

Process a custom research and writing request.

#### Required Request Body Schema

```json
{
  "query": "string (required)",
  "target_audience": "string (optional, default: 'general')",
  "max_sources": "integer (optional, default: 5, range: 1-20)",
  "style": "string (optional, default: 'informative')"
}
```

#### Examples

**Basic Request:**

```bash
curl -X POST "http://localhost:8000/agent/process" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Write about the future of renewable energy"
     }'
```

**Detailed Request:**

```bash
curl -X POST "http://localhost:8000/agent/process" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Explain quantum computing applications in cybersecurity",
       "target_audience": "IT professionals",
       "max_sources": 8,
       "style": "technical"
     }' | jq '.'
```

**Healthcare Topic:**

```bash
curl -X POST "http://localhost:8000/agent/process" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Write about telemedicine adoption post-COVID",
       "target_audience": "healthcare administrators",
       "max_sources": 10,
       "style": "analytical"
     }'
```

**Technology Topic:**

```bash
curl -X POST "http://localhost:8000/agent/process" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Impact of 5G technology on IoT devices",
       "target_audience": "engineers",
       "max_sources": 6,
       "style": "technical"
     }'
```

**Business Topic:**

```bash
curl -X POST "http://localhost:8000/agent/process" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Remote work productivity strategies",
       "target_audience": "business leaders",
       "max_sources": 7,
       "style": "informative"
     }'
```

**Extract Specific Data:**

```bash
# Get only the article
curl -s -X POST "http://localhost:8000/agent/process" \
     -H "Content-Type: application/json" \
     -d '{"query": "Machine learning in finance"}' | jq -r '.article'

# Get summary and key points
curl -s -X POST "http://localhost:8000/agent/process" \
     -H "Content-Type: application/json" \
     -d '{"query": "Blockchain in supply chain"}' | jq '{summary: .summary, key_points: .key_points}'

# Save article to file
curl -s -X POST "http://localhost:8000/agent/process" \
     -H "Content-Type: application/json" \
     -d '{"query": "Climate change solutions"}' | jq -r '.article' > article.txt
```

**Expected Response Structure:**

```json
{
  "status": "success",
  "topic": "extracted-topic",
  "article": "Generated article content...",
  "summary": "Article summary...",
  "sources": ["Source 1", "Source 2", "..."],
  "key_points": ["Key point 1", "Key point 2", "..."],
  "processing_time": 25.8,
  "agent_id": "uuid-string"
}
```

---

## API Documentation

### Interactive Documentation

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>

### OpenAPI Schema

```bash
# Get OpenAPI schema
curl -X GET "http://localhost:8000/openapi.json" | jq '.'
```

---

## Error Handling

### Common Error Responses

**Server Not Ready (500):**

```json
{
  "detail": "Coordinator not initialized"
}
```

**Invalid Request (422):**

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Processing Error (500):**

```json
{
  "detail": "Error message describing the issue"
}
```

### Test Error Scenarios

**Missing required field:**

```bash
curl -X POST "http://localhost:8000/agent/process" \
     -H "Content-Type: application/json" \
     -d '{"target_audience": "general"}'
```

**Invalid max_sources:**

```bash
curl -X POST "http://localhost:8000/agent/process" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Test query",
       "max_sources": 25
     }'
```

---

## Testing Scripts

### Quick Health Check

```bash
#!/bin/bash
echo "Testing API health..."
if curl -sf http://localhost:8000/health > /dev/null; then
    echo "✅ API is healthy"
else
    echo "❌ API is not responding"
fi
```

### Full API Test Suite

```bash
#!/bin/bash
BASE_URL="http://localhost:8000"

echo "🔍 Testing all API endpoints..."

# Health check
echo "1. Health check..."
curl -sf "${BASE_URL}/health" | jq '.status'

# Agent status
echo "2. Agent status..."
curl -sf "${BASE_URL}/agents/status" | jq '.coordinator_ready'

# Demo request
echo "3. Demo request..."
curl -sf -X POST "${BASE_URL}/agent/demo" | jq '.status'

# Custom request
echo "4. Custom request..."
curl -sf -X POST "${BASE_URL}/agent/process" \
     -H "Content-Type: application/json" \
     -d '{"query": "Test query"}' | jq '.status'

echo "✅ All tests completed"
```

---

## Performance Testing

### Response Time Testing

```bash
# Measure response time for health endpoint
curl -w "Health check: %{time_total}s\n" -s -X GET "http://localhost:8000/health" > /dev/null

# Measure response time for demo
curl -w "Demo request: %{time_total}s\n" -s -X POST "http://localhost:8000/agent/demo" > /dev/null

# Measure response time for custom request
curl -w "Custom request: %{time_total}s\n" -s -X POST "http://localhost:8000/agent/process" \
     -H "Content-Type: application/json" \
     -d '{"query": "Quick test"}' > /dev/null
```

### Load Testing with Multiple Requests

```bash
# Run 5 concurrent health checks
for i in {1..5}; do
    curl -s "http://localhost:8000/health" &
done
wait
echo "All health checks completed"
```

---

## Client Examples

### Python httpx Example

```python
import asyncio
import httpx

async def test_api():
    async with httpx.AsyncClient() as client:
        # Health check
        health = await client.get("http://localhost:8000/health")
        print(f"Health: {health.json()}")
        
        # Demo request
        demo = await client.post("http://localhost:8000/agent/demo")
        print(f"Demo status: {demo.json()['status']}")

asyncio.run(test_api())
```

### JavaScript fetch Example

```javascript
// Health check
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log('Health:', data));

// Custom request
fetch('http://localhost:8000/agent/process', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'Write about web development trends',
    target_audience: 'developers',
    max_sources: 5
  })
})
.then(response => response.json())
.then(data => console.log('Result:', data));
```

---

## Troubleshooting

### Common Issues

1. **Connection refused**: Make sure the server is running

   ```bash
   python start_api.py
   ```

2. **422 Validation Error**: Check request body format

   ```bash
   # Correct format
   curl -X POST "http://localhost:8000/agent/process" \
        -H "Content-Type: application/json" \
        -d '{"query": "Your query here"}'
   ```

3. **500 Internal Server Error**: Check server logs and environment variables

   ```bash
   # Make sure OPENAI_API_KEY is set
   echo $OPENAI_API_KEY
   ```

4. **Timeout**: Increase timeout for long-running requests

   ```bash
   curl --max-time 120 -X POST "http://localhost:8000/agent/demo"
   ```

### Debug Commands

```bash
# Check if server is listening
netstat -an | grep 8000

# Test with verbose output
curl -v -X GET "http://localhost:8000/health"

# Check server logs
tail -f dspy_multi_agent.log
```

---

## Summary

All endpoints are now ready for testing. The API provides:

- ✅ **Health monitoring** via `/health`
- ✅ **System status** via `/agents/status`  
- ✅ **Built-in demo** via `/agent/demo`
- ✅ **Custom requests** via `/agent/process`
- ✅ **Comprehensive documentation** via `/docs`

For automated testing, use the provided Python client (`demo_client.py`) or shell scripts (`test_demo_api.sh`).
