# Saarthi Government Services Backend

A Python FastAPI backend for the Saarthi chatbot that helps citizens access government services in local languages.

## Features

- **User Queries**: Handle natural language queries for government services
- **Service Information**: Fetch comprehensive details about government services
- **Mock Database**: Pre-loaded with information about common government services
- **JSON Responses**: All endpoints return structured JSON responses
- **Search Functionality**: Search services by keywords
- **Category Filtering**: Browse services by category
- **Error Handling**: Comprehensive error handling with meaningful responses

## Available Services

1. **Birth Certificate** - Apply for official birth registration
2. **Vehicle Registration** - Register motor vehicles
3. **Driving License** - Get authorization to operate vehicles
4. **Ration Card** - Access to subsidized food grains
5. **Passport** - Travel document for international travel

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/BargaviNarayanan/Saarthi.git
cd Saarthi/backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file (optional, for configuration):
```bash
cp .env.example .env
```

## Running the Server

### Development Mode
```bash
python main.py
```

Or with Uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Root & Health

- **GET** `/` - API information and available endpoints
- **GET** `/health` - Health check

### User Queries

- **POST** `/api/query` - Handle user queries for services
  ```json
  {
    "query": "apply for birth certificate",
    "language": "English"
  }
  ```

### Services

- **GET** `/api/services` - Get all available services
- **GET** `/api/services/{service_id}` - Get specific service details
  - Example: `/api/services/birth_certificate`
- **GET** `/api/services/category/{category}` - Get services by category
  - Example: `/api/services/category/transport`
- **GET** `/api/search?keyword=registration` - Search services by keyword

## API Examples

### 1. Query for Birth Certificate Service

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "apply for birth certificate",
    "language": "English"
  }'
```

**Response:**
```json
{
  "success": true,
  "query": "apply for birth certificate",
  "matched_services": [
    {
      "id": "birth_certificate",
      "name": "Birth Certificate",
      "description": "Official document issued to record the birth of a child",
      "category": "vital_records",
      "processing_time": "7-15 days",
      "cost": "Free",
      "required_documents": [...],
      "steps": [...],
      "eligibility": "Any child born in India",
      "authority": "Municipal Corporation / Local Registrar",
      "languages_supported": ["English", "Hindi", "Tamil", "Telugu", "Kannada"]
    }
  ],
  "message": "Found 1 service(s) matching your query.",
  "count": 1
}
```

### 2. Get All Services

```bash
curl "http://localhost:8000/api/services"
```

### 3. Get Specific Service Details

```bash
curl "http://localhost:8000/api/services/vehicle_registration"
```

### 4. Get Services by Category

```bash
curl "http://localhost:8000/api/services/category/transport"
```

### 5. Search Services

```bash
curl "http://localhost:8000/api/search?keyword=driving"
```

## Project Structure

```
backend/
├── main.py              # FastAPI application and endpoints
├── models.py            # Pydantic models for validation
├── database.py          # Mock database with service information
├── requirements.txt     # Python dependencies
├── .env.example         # Environment configuration template
└── README.md            # This file
```

## Data Models

### UserQuery
```python
{
  "query": str,           # User's query text
  "language": str,        # Language preference (default: "English")
  "user_id": str          # Optional user identifier
}
```

### ServiceResponse
```python
{
  "id": str,
  "name": str,
  "description": str,
  "category": str,
  "processing_time": str,
  "cost": str,
  "required_documents": List[str],
  "steps": List[str],
  "eligibility": str,
  "authority": str,
  "languages_supported": List[str]
}
```

## Future Enhancements

- [ ] Multi-language support (Hindi, Tamil, Telugu, Kannada, etc.)
- [ ] Integration with real government service databases
- [ ] NLP-based query understanding
- [ ] User authentication and tracking
- [ ] Service application form generation
- [ ] Real database integration (PostgreSQL/MongoDB)
- [ ] Caching for performance optimization
- [ ] API rate limiting
- [ ] Comprehensive logging and monitoring
- [ ] Automated testing

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Data validation with [Pydantic](https://pydantic-settings.readthedocs.io/)
- Server with [Uvicorn](https://www.uvicorn.org/)
