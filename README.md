# **FART-INLP**  
Repository for maintaining all code related to the **INLP Project**.

## **Training and Evaluation Logs**  
You can view the complete run details on Weights & Biases:  
[**View Run**](https://wandb.ai/aniruthzlatan-international-institue-of-information-tech/huggingface/runs/lwdoeupd)

---

## **Usage**  

Run the following commands to train and evaluate the model:

```bash
# Fine-tune the model and save the checkpoint after full training
python fine_tune_2.py  

# Evaluate the fine-tuned model (outputs accuracy, F1 score, and a sample prediction)
python test_finetuned_model.py  
```

----

## **Pretrained Weights**  
Fine-tuned **BERT** on the **SWAG dataset** for multiple-choice Q/A can be downloaded from:  
[**Download Weights**](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/aryan_garg_students_iiit_ac_in/EmNdN3AwkUdNmVYGxOkmv58Bi8FgAP-MyDX6RfOaKjnJtg?e=5X1kQl)

---

## **Results**  


![Q](results/q_a_test.png)

![R](results/final_result.png)
