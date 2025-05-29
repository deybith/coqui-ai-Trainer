from trainer.xtts.train_model import train_model

lang='es'
train_csv='/home/ubuntu/projects/create-av-content/models/salvador_mingo/dataset/metadata_train.csv'
eval_csv='/home/ubuntu/projects/create-av-content/models/salvador_mingo/dataset/metadata_eval.csv'
num_epochs=3
batch_size=16
grad_acumm=1
out_path='/home/ubuntu/projects/create-av-content/models/salvador_mingo'
max_audio_length=30

def main():
    train_model(
        lang,
        train_csv,
        eval_csv,
        num_epochs,
        batch_size,
        grad_acumm,
        out_path,
        max_audio_length
      )
                

if __name__ == "__main__":
    main()