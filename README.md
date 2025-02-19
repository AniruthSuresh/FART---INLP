# **FART-INLP**  
Repository for maintaining all code related to the **INLP Project**.

## **Training and Evaluation Logs**  
You can view the complete run details on Weights & Biases:  

Test 1 : Epoch 3 + Batch size = 8
[**View Run**](https://wandb.ai/aniruthzlatan-international-institue-of-information-tech/huggingface/runs/nng8h8kv?nw=nwuseraniruthzlatan)

---

## **Usage**  

Run the following commands to train and evaluate the model:

```bash
# Fine-tune the model and save the checkpoint after full training
python fnet_swag_fine_tuning.py  

# Evaluate the fine-tuned model (outputs accuracy, F1 score, and a sample prediction)
python test_finetuned_fnet_base.py  
```

----

## **Pretrained Weights**  
Fine-tuned **FNET** on the **SWAG dataset** for multiple-choice Q/A can be downloaded from:  

Test 1 : [**Download Weights**](https://wandb.ai/aniruthzlatan-international-institue-of-information-tech/huggingface/runs/nng8h8kv?nw=nwuseraniruthzlatan)

---

## **Results**  


Test 1 : ![Q](results/test_1_results.png) -> not the best results 

