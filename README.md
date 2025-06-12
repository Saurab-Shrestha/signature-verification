# Signature Verification System

A robust signature verification system that uses advanced computer vision and machine learning techniques to verify handwritten signatures. The system is designed to be resilient to variations in signature position, rotation, and scale while maintaining high accuracy in detecting forgeries.

## Features

- Advanced signature extraction using BiRefNet for background removal
- Rotation-invariant signature matching
- Multi-scale template matching
- Boundary point analysis
- Position-invariant feature comparison
- Web interface for easy signature verification
- Support for both genuine and forged signature detection

## Project Structure

```
signature_verification/
├── src/
│   ├── core/               # Core signature verification algorithms
│   ├── models/            # ML model definitions and weights
│   ├── utils/             # Utility functions
│   ├── static/            # Static files for web interface
│   ├── templates/         # HTML templates
│   ├── tests/             # Test cases
│   └── app.py             # Web application entry point
├── data/                  # Dataset directory
├── models/               # Trained model weights
├── temp/                 # Temporary files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/signature_verification.git
cd signature_verification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

1. Start the web server:
```bash
python src/app.py
```

2. Open your browser and navigate to `http://localhost:5000`

### Python API

```python
from src.core.signature_verification import compare_boundary_signatures

# Compare two signatures
results = compare_boundary_signatures(
    img1_path="path/to/signature1.png",
    img2_path="path/to/signature2.png"
)

# Get verification score
score = results['adjusted_score']
print(f"Verification score: {score:.2f}")
```

## Dependencies

- numpy >= 1.21.0
- opencv-python >= 4.5.0
- Pillow >= 8.0.0
- torch >= 1.9.0
- torchvision >= 0.10.0
- matplotlib >= 3.4.0
- scikit-image >= 0.18.0
- scipy >= 1.7.0
- kornia >= 0.6.0
- kornia-moons >= 0.2.0
- lightglue >= 0.1.0

## Development

### Running Tests

```bash
python -m pytest src/tests/
```

### Code Style

The project follows PEP 8 style guidelines. You can check your code style using:

```bash
flake8 src/
```

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- BiRefNet for background removal
- OpenCV for image processing
- PyTorch for deep learning capabilities 