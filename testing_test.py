from src.test import predict


preds = predict(
    dataFilePath="test_images/",
    bestModelPath="models/deployment/svm_model.joblib"
)

print(preds)