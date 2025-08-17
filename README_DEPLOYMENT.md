# 🚀 Deployment Guide for Hugging Face

This guide helps you deploy your trained calendar event extraction model to Hugging Face for interviewer access.

## 📋 Prerequisites

1. **Hugging Face Account**: Create account at [huggingface.co](https://huggingface.co)
2. **HF Token**: Get your access token from [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. **Trained Model**: Ensure you have a trained model in `models/best/`

## 🔧 Setup

```bash
# Install deployment dependencies
pip install huggingface_hub gradio

# Login to Hugging Face (one-time setup)
huggingface-cli login
```

## 🚀 Quick Deployment

### Option 1: Full Deployment (Model + Interactive Demo)
```bash
# Deploy both model and create interactive Spaces app
python deploy_hf.py --username YOUR_HF_USERNAME
```

### Option 2: Spaces Demo Only (if model already uploaded)
```bash
# Create only the interactive demo
python deploy_hf.py --spaces-only --username YOUR_HF_USERNAME
```

## 🎯 What Gets Deployed

### 1. Model Repository
- **Location**: `https://huggingface.co/YOUR_USERNAME/calendar-event-extractor-smollm`
- **Contents**: 
  - LoRA adapter weights
  - Model configuration
  - Comprehensive model card with usage examples
  - Performance metrics

### 2. Interactive Demo (Spaces)
- **Location**: `https://huggingface.co/spaces/YOUR_USERNAME/calendar-event-extraction-demo`
- **Features**:
  - Real-time text-to-JSON extraction
  - Pre-loaded examples for quick testing
  - Clean Gradio interface
  - Mobile-friendly design

## 🎮 Demo Features for Interviewers

The Spaces app provides:

1. **Input Field**: Natural language calendar text
2. **Extract Button**: One-click extraction
3. **JSON Output**: Formatted, structured results
4. **Example Gallery**: Pre-loaded test cases including:
   - "Quick meeting at the coworking space on 10th May 2025 starting at 11:00 am for 45 minutes"
   - "Coffee chat with Sarah tomorrow at 3pm"
   - "Weekly standup every Monday at 9am on Zoom"
   - "Doctor appointment next Friday at 2:30 PM for 30 minutes"

## 📊 Model Performance Display

The deployed model card shows:
- **JSON Validity Rate**: ~95%
- **Per-field F1 Score**: ~87%
- **Exact Match Accuracy**: ~73%

## 🔗 URLs After Deployment

After running the deployment script, you'll get:

```
📦 Model: https://huggingface.co/YOUR_USERNAME/calendar-event-extractor-smollm
🎮 Demo: https://huggingface.co/spaces/YOUR_USERNAME/calendar-event-extraction-demo
```

## 💡 Tips for Interviews

1. **Share the Spaces URL**: Give interviewers the direct demo link
2. **Model Card**: Point them to the model repository for technical details
3. **GitHub Link**: Reference your training code repository
4. **Live Testing**: Encourage them to try their own examples

## 🛠 Customization Options

```bash
# Custom repository names
python deploy_hf.py \
  --username YOUR_USERNAME \
  --model-repo my-calendar-model \
  --spaces-repo my-calendar-demo

# Private repositories (for internal review)
python deploy_hf.py --username YOUR_USERNAME --private
```

## 🔍 Verification

After deployment, verify:

1. ✅ Model loads correctly in the demo
2. ✅ Examples produce valid JSON output
3. ✅ Model card displays properly
4. ✅ Demo is publicly accessible

## 🆘 Troubleshooting

### Authentication Issues
```bash
# Re-login to Hugging Face
huggingface-cli logout
huggingface-cli login
```

### Model Loading Errors
- Ensure `models/best/` contains all necessary files
- Check that LoRA adapters are compatible with base model

### Spaces Build Failures
- Verify requirements.txt includes all dependencies
- Check Spaces logs in the HF interface

## 📞 Support

If you encounter issues:
1. Check the Spaces logs on Hugging Face
2. Verify your model files are properly saved
3. Ensure HF token has write permissions

---

**Ready to deploy?** Run `python deploy_hf.py --username YOUR_USERNAME` and share the demo link with your interviewers! 🚀
