
import argparse
import detect
import llm

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, 
					help='path to image')
parser.add_argument('--thresh', type=int, default=15, 
					help='threshold to distinguish new rows')
parser.add_argument('--order', type=str, default='yes', 
					help='enter y or yes to order detections in a human readable way')

args = parser.parse_args()

def transcript_processing_function(path, thresh, order):
    # path is the path to the transcript
    # transcript is png or pdf

    # Call the ocr to grab the text from the transcript (all text)
    text = ocr_processing(path, thresh, order)

    # ask llm to summarize the text in terms of metrics
    summary = llm_processing(text)
    return summary

def ocr_processing(path, thresh, order):
    predictions = detect.main(path, thresh, order)
    return predictions

def llm_processing(text):
    pipeline = llm.main()
    return pipeline(f"Provide the summary of the following:{text}")   

if __name__ == "__main__":
    summary = transcript_processing_function(
        args.image, args.thresh, args.order
    )
