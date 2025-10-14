# development notes for breathmetrics python
9/26/2025 scaffolding built. started building helper function. no test performed yet. 
9/27/2025 add find extrema. need to find a testing data 
9/29/2025 tested core functions. onset detection not working. need to rewrite the whole thing because chatgpt is being dumb 
9/30/2025 legacy onset detection added and tested 
10/1/2025 add primary core function added and working. behavior is slightly different from original breathmetrics? 
10/3/2025 added my pause detection method 
10/14/2025 add core logic and secondary methods. 

# pipeline from orig breathmetrics toolbox: 
## preproc
check input. i can take numpy array, CSV, .mat file. reshape to single vector 
optional input: crude inhale marker as a vec of sample number 
baseline correction. demean, zscore 
decide internally which resp to use. mean centered or not 
## feature extraction. all of these go into kernel.py 
### primary 
find extrema. *wrote, no test*
find onsets and pauses. this will have to be modified to new functions *wrote, no test*
find inhale and exhale offsets *wrote, no test*
find pause 
calculate inhale and exhale volume 
### secondary 
calculate secondary features 
calculate resp phase 
print secondary features (util?)
check if features are estimated (util?)
## ERPs and summary stats 
calculate ERP 
calculate resampled ERP 
## plotting 



---
# markdown syntex cheatsheet 
# Heading 1
## Heading 2
### Heading 3

*italic* or _italic_
**bold** or __bold__
***bold italic***
~~strikethrough~~

- Item 1
- Item 2
  - Nested item

1. First
2. Second
3. Third

[OpenAI](https://openai.com)

![Alt text](https://example.com/image.png)

Inline code: `print("hello")`

Block code:
```python
def greet():
    print("hello")

### Blockquotes
```markdown
> This is a quote

---



