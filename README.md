# README

## Installation

### Prerequisites

- Python 3.11 or higher
- Poetry (recommended) or pip

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/nilskujath/projetental.git
cd projetental

# Install dependencies
poetry install

# Activate the virtual environment
poetry env activate
```

Make sure to use `poetry run` before any command, e.g. `poetry run projetental --help`.

### Using pip

```bash
# Clone the repository
git clone git clone https://github.com/nilskujath/projetental.git
cd projetental

# Install in development mode
pip install -e .
```

## Data Requirements

### Raw Data Files (Required)

The system requires two input files in specific formats:

#### XML Data File
The XML file should contain word sense disambiguation instances with the following structure:
```xml
<corpus>
  <text id="...">
    <sentence id="...">
      <instance id="..." lemma="target_word" pos="...">
        <context>Sentence text with <head>target_word</head> marked</context>
      </instance>
    </sentence>
  </text>
</corpus>
```

**Key elements:**
- `<instance>`: Contains each disambiguation instance
- `lemma` attribute: The target word to disambiguate
- `<head>`: Marks the target word position in context
- `id` attribute: Unique identifier for each instance

#### Gold Standard Key File
The key file should contain sense annotations in the format:
```
instance_id sense_label
instance_id sense_label
...
```

**Format requirements:**
- Space-separated values
- First column: Instance ID (matching XML instance IDs)
- Second column: Sense label/identifier
- One instance per line

### Processed Data Files

- `processed_dataset.csv` - Main dataset with preprocessed text

**Note**: The `processed_dataset.csv` file can be automatically generated from the raw XML and key files using the data processing command.

### Data Processing

To create the processed dataset from raw XML and key files:

```bash
# Process raw files to create processed dataset
projetental --process-data <xml_file> <key_file> <output_file>

# Example with sample files:
projetental --process-data data/FSE-1.1.data.xml data/FSE-1.1.gold.key.txt data/processed_dataset.csv

# You can use any files that follow the required formats:
projetental --process-data my_corpus.xml my_annotations.txt my_dataset.csv
```



## Usage

### Command Line Interface

After installation, you can use the `projetental` command:

```bash
# Show help and available options
projetental --help

# Run specific experiments
projetental --supervised
projetental --unsupervised
projetental --semi-supervised

# Run multiple specific experiments
projetental --supervised --unsupervised

```

### Available Options

- `--process-data <xml_file> <key_file> <output_file>`: Process raw XML files to create processed dataset
- `--supervised`: Run supervised learning experiments
- `--unsupervised`: Run unsupervised clustering experiments
- `--semi-supervised`: Run semi-supervised clustering experiments (COP-KMeans)
- `--help`: Show detailed help message


## Results

All experiment results are automatically saved to the `results/` directory:

```
results/
├── supervised_results.csv          # Supervised learning results
├── unsup_oracle.csv               # Unsupervised with oracle k
├── unsup_estimated.csv            # Unsupervised with estimated k
└── semisup_all_clustering_results.csv # Semi-supervised clustering results
```

## Example Usage

```bash
# First time setup: process your XML and key files
projetental --process-data FSE-1.1.data.xml FSE-1.1.gold.key.txt processed_dataset.csv

# Run specific experiments
projetental --supervised
projetental --unsupervised
projetental --semi-supervised
```