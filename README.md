# keras-ocr Project

This is a project aiming to find the position (center coordinate) of a target word of a screenshot.

## Getting Started

### Installation
`keras-ocr` supports Python >= 3.6 and TensorFlow >= 2.0.0.
Install the packages according to requirements.txt.

1. python -m venv env 
2. .\env\Scripts\activate
3. pip install -r .\requirements.txt

### Usage
Input images or screenshots will need to be stored in the folder 'images'. 

The default target word is 'options'.

The script will draw the bounding boxes of all the detected words in green. And the target word will be framed in a blue bounding box. Output images will be stored in the folder 'test_output_keras'.

### API
1. Send a post request to /process as form-data
2. Include the screenshot as "file" and the word you are searching for as "word"
3. Will return a json response

```
{
    "result": "found",
    "x": 3464,
    "y": 1872
}

or 

{
    "result": "not found"
}
```