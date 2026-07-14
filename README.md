# ProPa

## Project Structure

```
UniDEC/
├── main.py                  # Entry point with Hydra configuration
├── trainer.py               # Training and evaluation runner
├── dense_clustering.py      # Dense embedding clustering
├── config.yaml              # Default configuration
├── models/
│   ├── unidec.py           # UniDEC model (dual encoder + classifier)
│   ├── supcon.py           # Supervised contrastive model
│   ├── llm.py              # LLM-based model with LoRA
│   ├── losses.py           # Shared loss computation
│   └── anns.py             # Approximate nearest neighbor search
├── data/
│   ├── datasets.py         # Dataset classes (QCDataset, XMLTestDataset, etc.)
│   ├── preprocessing.py    # Data loading and tokenization
│   ├── cluster.py          # Clustering utilities
│   └── tree.py             # Tree-based data structures
```

## Usage

### Basic Training

```bash
python main.py
```

### Custom Configuration

```bash
python main.py \
    data.dir=/path/to/data \
    data.dataset=LF-AmazonTitles-131K \
    model.encoder=UniDEC \
    model.pre_trained_model=distilbert \
    data.batch_size=1024 \
    model.learning_rate=1e-4 \
    model.num_epochs=150
```

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.encoder` | Model type (UniDEC, SupConDR, LLM) | UniDEC |
| `model.loss_lambda` | Weight for classification loss | 0.5 |
| `model.temperature` | Temperature for contrastive loss | 18.0 |
| `training.num_pos` | Number of positive labels per sample | 3 |
| `training.num_negs` | Number of hard negatives | 6 |
| `clustering.start_epoch` | Epoch to start batch clustering | 5 |

## Model Variants

- **UniDEC**: Full dual-encoder with both contrastive and classification heads
- **SupConDR**: Pure supervised contrastive learning approach
- **LLM**: Phi-3 based model with LoRA fine-tuning

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kharbanda2025unidec,
  title={UniDEC: Unified Dual Encoder and Classifier Training for Extreme Multi-Label Classification},
  author={Kharbanda, Siddhant and Gupta, Devaansh and K, Gururaj and Malhotra, Pankaj and Singh, Amit and Hsieh, Cho-Jui and Babbar, Rohit},
  booktitle={Proceedings of the ACM Web Conference 2025 (WWW '25)},
  year={2025},
  doi={10.1145/3696410.3714624}
}
```

## License

This project is licensed under the MIT License.
