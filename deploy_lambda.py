import boto3
import json
import os
import zipfile
import tempfile

# ── Config ─────────────────────────────────────────────────────────────────
ROLE_ARN       = "arn:aws:iam::448049787062:role/StockSenseAIRole"
BUCKET         = "stocksense-ai-448049787062"
REGION         = "us-east-1"
FUNCTION_NAME  = "stocksense-daily-retrain"
TICKERS        = "CRM,PLTR,NVDA"

lambda_client  = boto3.client("lambda",      region_name=REGION)
events_client  = boto3.client("events",      region_name=REGION)
sns_client     = boto3.client("sns",         region_name=REGION)

# ── Step 1: Create SNS topic for alerts ────────────────────────────────────
print("Step 1: Creating SNS alert topic...")
sns_response = sns_client.create_topic(Name="stocksense-alerts")
SNS_ARN      = sns_response["TopicArn"]
print(f"SNS topic: {SNS_ARN}")

# Subscribe your email — change this to your email
EMAIL = "ankur.vyas998@gmail.com"   # ← CHANGE THIS
sns_client.subscribe(
    TopicArn = SNS_ARN,
    Protocol = "email",
    Endpoint = EMAIL
)
print(f"Subscription confirmation sent to {EMAIL}")
print("Check your email and confirm the subscription before proceeding.")

# ── Step 2: Package Lambda function ────────────────────────────────────────
print("\nStep 2: Packaging Lambda function...")
zip_path = "/tmp/stocksense_lambda.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write("lambda/retrain_handler.py", "retrain_handler.py")
print(f"Lambda package created: {zip_path}")

# ── Step 3: Create or update Lambda function ───────────────────────────────
print("\nStep 3: Deploying Lambda function...")
with open(zip_path, "rb") as f:
    zip_bytes = f.read()

try:
    # Try creating new function
    response = lambda_client.create_function(
        FunctionName  = FUNCTION_NAME,
        Runtime       = "python3.11",
        Role          = ROLE_ARN,
        Handler       = "retrain_handler.lambda_handler",
        Code          = {"ZipFile": zip_bytes},
        Timeout       = 900,    # 15 minutes max
        MemorySize    = 1024,   # 1GB RAM for ML training
        Environment   = {
            "Variables": {
                "BUCKET"             : BUCKET,
                "ROLE_ARN"           : ROLE_ARN,
                "REGION"             : REGION,
                "TICKERS"            : TICKERS,
                "ACCURACY_THRESHOLD" : "0.55",
                "SNS_ARN"            : SNS_ARN,
            }
        },
        Description = "StockSense AI — daily model retraining pipeline"
    )
    FUNCTION_ARN = response["FunctionArn"]
    print(f"Lambda created: {FUNCTION_ARN}")

except lambda_client.exceptions.ResourceConflictException:
    # Function exists — update it
    lambda_client.update_function_code(
        FunctionName = FUNCTION_NAME,
        ZipFile      = zip_bytes
    )
    lambda_client.update_function_configuration(
        FunctionName = FUNCTION_NAME,
        Timeout      = 900,
        MemorySize   = 1024,
        Environment  = {
            "Variables": {
                "BUCKET"             : BUCKET,
                "ROLE_ARN"           : ROLE_ARN,
                "REGION"             : REGION,
                "TICKERS"            : TICKERS,
                "ACCURACY_THRESHOLD" : "0.55",
                "SNS_ARN"            : SNS_ARN,
            }
        }
    )
    response     = lambda_client.get_function(FunctionName=FUNCTION_NAME)
    FUNCTION_ARN = response["Configuration"]["FunctionArn"]
    print(f"Lambda updated: {FUNCTION_ARN}")

# ── Step 4: Create EventBridge rule — fires daily at 4:30pm ET ────────────
print("\nStep 4: Creating EventBridge daily trigger...")
# 4:30pm ET = 21:30 UTC
rule_response = events_client.put_rule(
    Name                = "stocksense-daily-retrain",
    ScheduleExpression  = "cron(30 21 ? * MON-FRI *)",  # weekdays only
    State               = "ENABLED",
    Description         = "Triggers StockSense AI retraining at market close"
)
RULE_ARN = rule_response["RuleArn"]
print(f"EventBridge rule: {RULE_ARN}")

# ── Step 5: Give EventBridge permission to invoke Lambda ───────────────────
print("\nStep 5: Adding EventBridge → Lambda permission...")
try:
    lambda_client.add_permission(
        FunctionName  = FUNCTION_NAME,
        StatementId   = "EventBridgeDailyRetrain",
        Action        = "lambda:InvokeFunction",
        Principal     = "events.amazonaws.com",
        SourceArn     = RULE_ARN,
    )
    print("Permission added")
except lambda_client.exceptions.ResourceConflictException:
    print("Permission already exists")

# ── Step 6: Connect EventBridge rule to Lambda ─────────────────────────────
print("\nStep 6: Connecting EventBridge to Lambda...")
events_client.put_targets(
    Rule    = "stocksense-daily-retrain",
    Targets = [{
        "Id"  : "StockSenseLambda",
        "Arn" : FUNCTION_ARN,
    }]
)
print("EventBridge → Lambda connected")

# ── Step 7: Test invoke immediately ───────────────────────────────────────
print("\nStep 7: Test invoking Lambda now...")
print("This will train all models and upload to S3 — takes ~3 minutes...")
test_response = lambda_client.invoke(
    FunctionName   = FUNCTION_NAME,
    InvocationType = "RequestResponse",  # wait for result
    LogType        = "Tail",
)

result = json.loads(test_response["Payload"].read())
print(f"\nLambda response:")
print(json.dumps(result, indent=2))

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"StockSense AI Pipeline Deployed")
print(f"{'='*55}")
print(f"Lambda function : {FUNCTION_NAME}")
print(f"Schedule        : Weekdays at 4:30pm ET")
print(f"Tickers         : {TICKERS}")
print(f"Alert email     : {EMAIL}")
print(f"S3 artifacts    : s3://{BUCKET}/model-artifacts/")
print(f"SNS alerts      : {SNS_ARN}")
print(f"\nView in AWS Console:")
print(f"  Lambda   → https://us-east-1.console.aws.amazon.com/lambda")
print(f"  EventBridge → https://us-east-1.console.aws.amazon.com/events")
print(f"  S3       → https://s3.console.aws.amazon.com/s3/buckets/{BUCKET}")