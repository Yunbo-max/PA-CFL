# PA-CFL: Privacy-Adaptive Clustered Federated Learning for Transformer-Based Sales Forecasting

[![arXiv](https://img.shields.io/badge/arXiv-2503.12220-b31b1b.svg)](https://arxiv.org/abs/2503.12220)

PA-CFL (*Privacy-Adaptive Clustered Federated Learning*) is a federated learning framework for demand forecasting in retail environments with heterogeneous data. It groups retailers into privacy-aware clusters ("bubbles") using differential privacy and feature importance, then trains Transformer models for localized sales prediction.

## Key Features
- **Heterogeneity-Aware Clustering**: Groups retailers by feature importance to isolate data discrepancies
- **Adaptive Differential Privacy**: Dynamically adjusts noise levels per cluster
- **Poisoning Defense**: Filters malicious clients via risk assessment
- **Transformer-Based Forecasting**: Self-attention models for accurate predictions

## Performance
| Metric        | Improvement vs Local Learning |
|---------------|------------------------------|
| R² Score      | +5.4%                        |
| RMSE          | -69%                         |
| MAE           | -45%                         |

## Installation&Usage
```bash
git clone https://github.com/your-repo/PA-CFL.git
cd PA-CFL
pip install -r requirements.txt
bash PA-CFL/Model_training/original/run.sh
```
## Citation
@article{long2025bubble,
  title={PA-CFL: Privacy-Adaptive Clustered Federated Learning for Transformer-Based Sales Forecasting on Heterogeneous Retail Data},
  author={Long, Yunbo and Xu, Liming and Zheng, Ge and Brintrup, Alexandra},
  journal={arXiv preprint arXiv:2503.12220},
  year={2025}
}
