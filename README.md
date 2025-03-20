# **FART-INLP**  
Repository for maintaining all code related to the **INLP Project**.

## **Training and Evaluation Logs**  
You can view the complete run details on Weights & Biases:  

[**View Run**](https://wandb.ai/aniruthzlatan-international-institue-of-information-tech/jedi-configs/runs/xhn7f79x?nw=nwuseraniruthzlatan)

---

## **Usage**  

Run the following commands to train and evaluate the model:

```bash
# Trains the model and saves the weights and tokenizer 
python BART_finetune_training.py 

# Evaluate the fine-tuned model 
python BART_BERTSCORE_eval.py

```

----

## **Pretrained Weights**  

Test 1 : [**Download Weights**](https://wandb.ai/aniruthzlatan-international-institue-of-information-tech/huggingface/runs/nng8h8kv?nw=nwuseraniruthzlatan)



---

## **Results**  
It uses both the encoder and the decoder and the results of evaluation of base - BART on these metrics are as shown : 

1. BART-BERT SCORE : 



Test 1 : ![Q](results/test_1_results.png) -> not the best results 



