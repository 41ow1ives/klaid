import wandb

def wandb_init(fact_model, law_model, args):
    wandb.init(
        project=args.project,
        entity=args.entity
    )

    # wandb에 기록하고 싶은 정보는 json에서 가져다 update로 추가해줄 수 있다.
    wandb.config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.num_epochs,
        "optimizer": args.optimizer,
    }
    
    wandb.watch((fact_model, law_model))