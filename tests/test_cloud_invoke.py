import base64
import io
import json

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber

from bikeml.cloud import Cloud, parse_report

REPORT = ("START RequestId: abc Version: 3\n"
          "END RequestId: abc\n"
          "REPORT RequestId: abc\tDuration: 812.34 ms\tBilled Duration: 813 ms\tMemory Size: 1024 MB\t"
          "Max Memory Used: 187 MB\tInit Duration: 2950.12 ms\t\n")


def test_parse_report():
    r = parse_report(REPORT)
    assert r == {"duration_ms": 812.34, "billed_ms": 813.0, "memory_mb": 1024.0,
                 "max_memory_used_mb": 187.0, "init_ms": 2950.12}
    assert parse_report("Task timed out after 3.00 seconds")["timed_out"] is True


def fake_cloud():
    c = Cloud.__new__(Cloud)
    c.fn = "bike-demand-predict"
    c.lam = boto3.client("lambda", region_name="ap-northeast-2",
                         aws_access_key_id="x", aws_secret_access_key="y")
    return c


def invoke_with(response_payload, function_error=None, log=REPORT):
    c = fake_cloud()
    data = json.dumps(response_payload).encode()
    resp = {"StatusCode": 200, "ExecutedVersion": "3", "Payload": StreamingBody(io.BytesIO(data), len(data)),
            "LogResult": base64.b64encode(log.encode()).decode()}
    if function_error:
        resp["FunctionError"] = function_error
    with Stubber(c.lam) as st:
        st.add_response("invoke", resp, {"FunctionName": c.fn, "Qualifier": "3",
                                         "Payload": json.dumps({"instances": []}).encode(), "LogType": "Tail"})
        return c.invoke({"instances": []}, "3")


def test_invoke_ok():
    body, meta = invoke_with({"statusCode": 400, "body": {"ok": False, "error": {"code": "INVALID_INPUT"}}})
    assert body["statusCode"] == 400 and meta["function_version"] == "3"
    assert meta["report"]["max_memory_used_mb"] == 187.0 and meta["function_error"] is None


def test_invoke_unhandled_error_is_failure():
    body, meta = invoke_with({"errorMessage": "Task timed out after 3.00 seconds", "errorType": "Sandbox.Timedout"},
                             function_error="Unhandled", log="Task timed out after 3.00 seconds\n")
    assert body["statusCode"] is None and body["body"]["error"]["code"] == "FUNCTION_ERROR"
    assert meta["report"]["timed_out"] and "log_tail" in meta


def test_invoke_non_object_response():
    body, _ = invoke_with(None)
    assert body["statusCode"] is None and body["body"]["error"]["code"] == "BAD_RESPONSE"
