# Brain MRI Assistant

**Live demo:** https://brain-tumor-henna.vercel.app/

Upload a brain MRI image. The app tells which of 4 classes it looks like (glioma, meningioma, pituitary tumor, no tumor), shows a heatmap of the area the model focused on, and an LLM explains the result in simple words with the type of doctor to visit. The LLM is not allowed to mention any medicine.

## Folder layout
    api/index.py       FastAPI backend (ONNX inference, heatmap, LLM report)
    public/index.html  Web page (upload, results, heatmap, report)
    model/             model.onnx (8.6 MB), cam_weights.npy, classes.json
    train/train.py     Training + ONNX export script
    vercel.json  requirements.txt  .env.example

## Model
MobileNetV2 (ImageNet pretrained), fine-tuned at 224x224 on the Kaggle "Brain Tumor Classification (MRI)" dataset (Sartaj Bhuvaji, 3264 images).
The heatmap is a Class Activation Map (CAM), computed from the last feature maps and classifier weights.

Honest results:
- Strict run (train on Training folder, test on the original Testing folder): 79.9% accuracy, glioma recall only 35%. The Testing folder looks different from Training.
- Shipped model (half of Testing mixed into training, evaluated on the other half, 198 images): 90.4% accuracy. Recall: glioma 64%, meningioma 100%, pituitary 97%, no tumor 100%.
  Slices from the same patient may appear on both sides, so real world accuracy will be lower. Glioma is the weakest class.

## Run locally
    pip install -r requirements.txt uvicorn
    cp .env.example .env   # fill in GROQ_API_KEY; optional, without it a basic template report is shown
    uvicorn api.index:app --port 8000
Then open http://localhost:8000 in your browser (the same server serves the page and the API).

## Deploy on Vercel
1. Push this folder to a GitHub repo (or run `npx vercel` inside it).
2. Import the repo in Vercel. No build command needed.
3. Add environment variables: GROQ_API_KEY, and optionally LLM_MODEL (default openai/gpt-oss-120b).
4. Deploy, then open the URL and upload a scan.

## Retrain
    python train/train.py --data path/to/dataset --epochs 7 --mix_test
Needs torch, torchvision, timm, onnx. Put ImageNet MobileNetV2 (timm) weights at train/mobilenetv2_100_pretrained.pth. The script exports to model/.

## Limits
Educational demo, not a medical device. Trained on MRI only, so CT and X-ray images give meaningless results. It only knows 3 tumor types, not other brain diseases (stroke, bleeding, infection).
