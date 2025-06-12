# Signature Verification System

A comprehensive signature verification system that uses multiple approaches to verify the authenticity of signatures. The system combines traditional image processing techniques with deep learning models to provide robust signature verification.

## Features

- Multiple signature cleaning methods:
  - Advanced cleaning with adaptive thresholding
  - Denoising-based cleaning
  - Watershed-based cleaning for better component separation

- Feature extraction:
  - Boundary point detection and analysis
  - Texture feature extraction using GLCM
  - Gradient analysis
  - Connected component analysis

- Multiple verification approaches:
  - Boundary-based comparison
  - Texture-based comparison
  - Deep learning model-based comparison (LoFTR, DeDoDe, LightGlue)
  - Combined approach using all methods

- Visualization tools for analysis and debugging

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd signature_verification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from src.core.signature_verification import (
    clean_signature_advanced,
    compare_signatures_combined,
    visualize_boundary_results
)

# Clean signatures
clean_signature_advanced('path/to/signature1.jpg', 'path/to/cleaned1.jpg')
clean_signature_advanced('path/to/signature2.jpg', 'path/to/cleaned2.jpg')

# Compare signatures
results = compare_signatures_combined('path/to/cleaned1.jpg', 'path/to/cleaned2.jpg')

# Visualize results
fig = visualize_boundary_results(results)
fig.show()
```

## Project Structure

```
src/core/signature_verification/
├── preprocessing/
│   └── cleaning.py         # Signature cleaning functions
├── features/
│   └── extraction.py       # Feature extraction functions
├── matching/
│   └── comparison.py       # Signature comparison functions
├── models/
│   └── cyclegan.py         # Deep learning model definitions
├── utils/
│   └── visualization.py    # Visualization utilities
└── __init__.py            # Main module exports
```

## Dependencies

- numpy>=1.21.0
- opencv-python>=4.5.0
- Pillow>=8.0.0
- torch>=1.9.0
- torchvision>=0.10.0
- matplotlib>=3.4.0
- scikit-image>=0.18.0
- scipy>=1.7.0
- kornia>=0.6.0
- kornia-moons>=0.2.0
- lightglue>=0.1.0

## License

[Specify your license here]

## Contributing

[Add contribution guidelines here] 