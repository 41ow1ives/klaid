import argparse

def parse_default_args(parser):
    # Setting related
    parser.add_argument("--seed", default=417)

    # BERT related
    parser.add_argument("--model_version", default='bert-base-multilingual-cased', type=str)

    # Data related
    parser.add_argument("--label_path", default='./data/klaid_label.csv', type=str)

    # Training related
    parser.add_argument("--gpu_num", default=1, type=int)
    parser.add_argument("--test_size", default=0.3, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--step_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-08, type=float)
    parser.add_argument("--gamma", default=0.8, type=float)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument("--adam_eps", default=1e-08, type=float)
    parser.add_argument("--num_accumulation_step", default=1, type=int)
    
    # Save related
    parser.add_argument("--model_dir", default="./models", type=str)
    
    # Wandb related
    parser.add_argument("--project", default="KLAID", type=str)
    parser.add_argument("--entity", default="77601251")
    
    args = parser.parse_args()
    return args