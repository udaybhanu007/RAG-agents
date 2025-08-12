# Development Environment Configuration

This project is configured for development with a single environment setup using `.env.dev`.

## Environment Files

### 📁 Available Environment Files

- **`.env`** - Default environment configuration (fallback)
- **`.env.dev`** - Development environment configuration (primary)

### 🔧 Development Configuration Features

| Setting | Value | Description |
|---------|-------|-------------|
| **Log Level** | DEBUG | Detailed logging for development |
| **Debug Mode** | true | Enables debug features |
| **JSON Logs** | false | Human-readable console output |
| **Colored Logs** | true | Enhanced readability |
| **API Host** | localhost | Local development server |
| **Collections** | *_dev suffix | Separate dev data |
| **Containers** | *_dev suffix | Isolated dev storage |
| **Cache** | false | Disabled for development |
| **Verbose Logging** | true | Extra detailed output |

## 🚀 Usage

The application automatically loads the development configuration:

```bash
# Simply run your application - .env.dev loads automatically
python src/core/util.py
```

### In Code

```python
from src.core.env_config import load_environment_config, print_environment_info

# Load development config (default)
load_environment_config()

# Print environment information
print_environment_info()
```

## 📋 Environment Information

Use the environment configuration utility to see current settings:

```python
from src.core.env_config import print_environment_info

# Print detailed environment info
print_environment_info()
```

Output example:
```
==================================================
🛠️  DEVELOPMENT ENVIRONMENT
==================================================
Environment: DEVELOPMENT
Debug Mode: true
Log Level: DEBUG
API Host: localhost
API Port: 8000
Development Mode: true
==================================================
```

## 🔐 Security Best Practices

1. **API Keys**: Replace placeholder values with actual keys
2. **Key Vault**: Store sensitive credentials in Azure Key Vault
3. **Development Data**: Use separate dev collections/containers
4. **Git Security**: Never commit real API keys

## 📝 Configuration Files

- **`.env.dev`**: Main development configuration
- **`.env`**: Fallback configuration (if .env.dev missing)

## 🐛 Troubleshooting

- **Configuration not loading**: Check `.env.dev` file exists
- **Wrong settings**: Verify values in `.env.dev`
- **Import errors**: Ensure `src.core.env_config` is accessible
