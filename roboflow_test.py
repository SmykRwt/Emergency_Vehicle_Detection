from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="1rjU14yv8zWaiGEetWyY"
)

result = client.run_workflow(
    workspace_name="samyak-cb3dw",
    workflow_id="detect-count-and-visualize-7",
    images={
        "image": "YOUR_IMAGE.jpg"
    },
    use_cache=True # cache workflow definition for 15 minutes
)
