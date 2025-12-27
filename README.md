# A Recipe Generator for Inorganic Solid-Phase Materials Synthesis
To be update.
* Cite: (To be update)      
* Dataset: [https://huggingface.co/datasets/wjsehdrnfl428/dataset_for_recipe_generator](https://huggingface.co/datasets/wjsehdrnfl428/dataset_for_recipe_generator)
---

## Usage 
## Local environment: 

Use python=3.11 and install the required packages using the following command:     
```
pip install -r requirements.txt
```
Run in your environment to predict the synthesis recipe.     
```
streamlit run demo.py
```


1. Enter your personal ```OpenAI``` API key.                
        We added the following model options to the recipe generator.
   ```
   model_options = [
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "gpt-5.2",
        "gpt-5-mini",
        "gpt-5",
        "o3",
        "o3-mini",
        "o3-mini-low",
        "o3-mini-high"
    ]
    ```
    Make sure your API key has access to the model you want to use. 
    
3. Click the ```Update``` key button.
4. Provide the information needed for prediction (e.g., materials, synthesis technique, application, etc.).
5. (Optional) Adjust the number of data entries used for RAG, or upload additional reference papers (PDFs).
6. Click ```Recommend``` button at the bottom left to generate a recipe
7. Click ```Clear Conversation``` to start a new prediction, or continue with follow-up Q&A using the ```chat box``` below the generated response. 


        

## Or try it on our website:     
[https://ssr.recipe-generator.site/](https://ssr.recipe-generator.site/)        
    ***(The API key is used only for your request and is not saved or tracked. See ```demo.py``` for details.)***


## Contact info
* e-mail: jdwjyl2007@ajou.ac.kr
