# Vision-RAG Waste Management

AI/ML system for waste image classification combined with
Retrieval-Augmented Generation (RAG) for grounded waste-management
guidance.

## Classes

- Paper Waste
- Organic Waste
- Leather Waste
- Plastic Waste
- Glass Waste

## Pipeline

Image + Question
        ↓
Waste Image Classifier
        ↓
Predicted Class + Confidence
        ↓
RAG Retrieval
        ↓
Grounded Answer + Citations
