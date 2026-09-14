using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace AiXrPortfolio
{
    public class XrAssistant : MonoBehaviour
    {
        public string serverUrl = "http://127.0.0.1:8000";
        public string apiKey = "";
        public InputField input;
        public Text output;
        public Text status;
        public Transform demoCube;
        public AudioSource speaker;
        public Button confirmButton;
        private string sessionId;
        private string pendingActionId;
        private string lastAnswer;
        private bool busy;

        [Serializable] public class SceneState { public string target = "demo_cube"; public float rotation_y; }
        [Serializable] public class ChatRequest { public string message; public string session_id; public SceneState scene; }
        [Serializable] public class Action { public string action; public string target; public float value; }
        [Serializable] public class Pending { public string action_id; public Action action; }
        [Serializable] public class Reply { public string answer; public string mode; public Pending[] pending_actions; }
        [Serializable] public class Proposal { public string session_id; public string action; public string target = "demo_cube"; public float value; }
        [Serializable] public class Confirmation { public string session_id; public string action_id; }
        [Serializable] public class Confirmed { public string status; public Action action; }
        [Serializable] public class SpeechRequest { public string text; }

        void Start()
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; i++)
                if (args[i] == "-server-url") serverUrl = args[i + 1];
            sessionId = Guid.NewGuid().ToString("N");
            confirmButton.interactable = false;
            status.text = "Backend: " + serverUrl;
            if (Array.IndexOf(args, "-portfolio-smoke") >= 0) StartCoroutine(SmokeTest());
        }

        IEnumerator SmokeTest()
        {
            yield return null;
            float initial = demoCube.eulerAngles.y;
            Propose("rotate");
            yield return new WaitUntil(() => !busy);
            if (string.IsNullOrEmpty(pendingActionId)) { Debug.LogError("AI_XR_SMOKE_FAILED proposal"); Application.Quit(2); yield break; }
            bool unchangedBeforeConfirm = Mathf.Abs(Mathf.DeltaAngle(initial, demoCube.eulerAngles.y)) < .01f;
            Confirm();
            yield return new WaitUntil(() => !busy);
            float rotation = Mathf.DeltaAngle(initial, demoCube.eulerAngles.y);
            bool ok = unchangedBeforeConfirm && Mathf.Abs(rotation - 45) < .01f && pendingActionId == null;
            Debug.Log("AI_XR_SMOKE " + JsonUtility.ToJson(new SmokeResult { success = ok, rotation = rotation, unchanged_before_confirmation = unchangedBeforeConfirm }));
            Application.Quit(ok ? 0 : 3);
        }

        [Serializable] public class SmokeResult { public bool success; public float rotation; public bool unchanged_before_confirmation; }

        public void Send()
        {
            if (busy || string.IsNullOrWhiteSpace(input.text)) return;
            var request = new ChatRequest { message = input.text, session_id = sessionId,
                scene = new SceneState { rotation_y = demoCube.eulerAngles.y } };
            StartCoroutine(Post("/agent/chat", JsonUtility.ToJson(request), body =>
            {
                var reply = JsonUtility.FromJson<Reply>(body);
                lastAnswer = reply.answer;
                output.text = reply.answer;
                status.text = "Provider: " + reply.mode;
                if (reply.pending_actions != null && reply.pending_actions.Length > 0)
                {
                    pendingActionId = reply.pending_actions[0].action_id;
                    confirmButton.interactable = true;
                }
            }));
        }

        public void ProposeRotate() { Propose("rotate"); }

        public void Propose(string kind)
        {
            if (busy) return;
            StartCoroutine(Post("/agent/propose", JsonUtility.ToJson(new Proposal {
                session_id = sessionId, action = kind, value = 45 }), body =>
            {
                var action = JsonUtility.FromJson<Pending>(body);
                pendingActionId = action.action_id;
                confirmButton.interactable = true;
                output.text = "Proposed " + kind + ". Select Confirm to execute.";
            }));
        }

        public void Confirm()
        {
            if (busy || string.IsNullOrEmpty(pendingActionId)) return;
            string actionId = pendingActionId;
            confirmButton.interactable = false;
            StartCoroutine(Post("/agent/confirm", JsonUtility.ToJson(new Confirmation {
                session_id = sessionId, action_id = actionId }), body =>
            {
                var reply = JsonUtility.FromJson<Confirmed>(body);
                if (reply.status != "confirmed" || reply.action == null || reply.action.target != "demo_cube") return;
                switch (reply.action.action)
                {
                    case "rotate": demoCube.Rotate(0, Mathf.Clamp(reply.action.value, -360, 360), 0); break;
                    case "highlight": demoCube.GetComponent<Renderer>().material.color = new Color(.35f, .95f, .8f); break;
                    case "reset": demoCube.rotation = Quaternion.identity; demoCube.GetComponent<Renderer>().material.color = Color.gray; break;
                    case "show_step": output.text = "Training demonstration step " + reply.action.value; break;
                }
                pendingActionId = null;
                status.text = "Confirmed action executed locally";
            }));
        }

        public void Speak()
        {
            if (!busy && !string.IsNullOrEmpty(lastAnswer)) StartCoroutine(Synthesize(lastAnswer));
        }

        IEnumerator Synthesize(string answer)
        {
            busy = true;
            using (var request = new UnityWebRequest(serverUrl + "/speech/synthesize", "POST"))
            {
                request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(JsonUtility.ToJson(new SpeechRequest { text = answer })));
                request.downloadHandler = new DownloadHandlerAudioClip(serverUrl + "/speech/synthesize", AudioType.WAV);
                request.SetRequestHeader("Content-Type", "application/json");
                if (!string.IsNullOrEmpty(apiKey)) request.SetRequestHeader("X-API-Key", apiKey);
                request.timeout = 90;
                yield return request.SendWebRequest();
                if (request.result == UnityWebRequest.Result.Success)
                { speaker.clip = DownloadHandlerAudioClip.GetContent(request); speaker.Play(); }
                else status.text = "Speech unavailable: " + request.error;
            }
            busy = false;
        }

        IEnumerator Post(string path, string json, System.Action<string> onSuccess)
        {
            busy = true;
            status.text = "Waiting for AI service...";
            using (var request = new UnityWebRequest(serverUrl.TrimEnd('/') + path, "POST"))
            {
                request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                if (!string.IsNullOrEmpty(apiKey)) request.SetRequestHeader("X-API-Key", apiKey);
                request.timeout = 120;
                yield return request.SendWebRequest();
                if (request.result == UnityWebRequest.Result.Success)
                {
                    try { onSuccess(request.downloadHandler.text); }
                    catch (Exception ex) { status.text = "Invalid server response: " + ex.Message; }
                }
                else status.text = "Request failed: " + request.responseCode + " " + request.error;
            }
            busy = false;
        }
    }
}
