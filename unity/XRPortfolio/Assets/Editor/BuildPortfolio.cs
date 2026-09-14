using System;
using System.IO;
using AiXrPortfolio;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public static class BuildPortfolio
{
    static Font Font { get { return Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf"); } }

    static RectTransform Rect(GameObject go, Transform parent, Vector2 position, Vector2 size)
    {
        var rect = go.GetComponent<RectTransform>() ?? go.AddComponent<RectTransform>();
        rect.SetParent(parent, false);
        rect.anchorMin = rect.anchorMax = new Vector2(0, 1);
        rect.pivot = new Vector2(0, 1);
        rect.anchoredPosition = position;
        rect.sizeDelta = size;
        return rect;
    }

    static Text Label(string name, Transform parent, string value, Vector2 pos, Vector2 size, int fontSize=20)
    {
        var go = new GameObject(name, typeof(RectTransform), typeof(Text));
        Rect(go, parent, pos, size);
        var label = go.GetComponent<Text>();
        label.text = value; label.font = Font; label.fontSize = fontSize; label.color = Color.white;
        label.horizontalOverflow = HorizontalWrapMode.Wrap;
        label.verticalOverflow = VerticalWrapMode.Truncate;
        return label;
    }

    static Button Button(string name, Transform parent, Vector2 pos, UnityEngine.Events.UnityAction action)
    {
        var go = new GameObject(name, typeof(RectTransform), typeof(Image), typeof(Button));
        Rect(go, parent, pos, new Vector2(165, 40));
        go.GetComponent<Image>().color = new Color(.16f, .28f, .36f);
        var label = Label("Label", go.transform, name, Vector2.zero, new Vector2(165, 40), 17);
        label.alignment = TextAnchor.MiddleCenter;
        var button = go.GetComponent<Button>();
        UnityEditor.Events.UnityEventTools.AddPersistentListener(button.onClick, action);
        return button;
    }

    [MenuItem("AI XR/Generate Demo Scene")]
    public static void GenerateScene()
    {
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        var cam = new GameObject("Main Camera", typeof(Camera), typeof(AudioListener));
        cam.tag = "MainCamera";
        cam.transform.position = new Vector3(3, 2, -5);
        cam.transform.LookAt(Vector3.zero);
        cam.GetComponent<Camera>().backgroundColor = new Color(.06f, .09f, .13f);
        cam.GetComponent<Camera>().clearFlags = CameraClearFlags.SolidColor;
        var light = new GameObject("Directional Light", typeof(Light));
        light.GetComponent<Light>().type = LightType.Directional;
        light.transform.rotation = Quaternion.Euler(40, -30, 0);
        var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.name = "demo_cube";
        cube.transform.position = new Vector3(2.7f, 0, 0);
        Directory.CreateDirectory("Assets/Generated");
        var material = AssetDatabase.LoadAssetAtPath<Material>("Assets/Generated/Cube.mat");
        if (material == null) {
            material = new Material(Shader.Find("Standard"));
            AssetDatabase.CreateAsset(material, "Assets/Generated/Cube.mat");
        }
        material.color = Color.gray;
        cube.GetComponent<Renderer>().sharedMaterial = material;
        var canvasGo = new GameObject("Canvas", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        canvasGo.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
        var scaler = canvasGo.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1280, 800);
        new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
        var manager = new GameObject("XR Assistant", typeof(XrAssistant), typeof(AudioSource));
        var app = manager.GetComponent<XrAssistant>();
        app.demoCube = cube.transform; app.speaker = manager.GetComponent<AudioSource>();
        Label("Title", canvasGo.transform, "AI XR Portfolio", new Vector2(30,-25), new Vector2(600,55), 32);
        Label("Subtitle", canvasGo.transform, "Local RAG + tool calling + confirmed scene actions", new Vector2(30,-85), new Vector2(650,40), 18);
        var field = new GameObject("Input", typeof(RectTransform), typeof(Image), typeof(InputField));
        Rect(field, canvasGo.transform, new Vector2(30,-150), new Vector2(690,100));
        field.GetComponent<Image>().color = new Color(.1f,.15f,.21f);
        var value = Label("Text",field.transform,"",new Vector2(12,-10),new Vector2(660,80),19);
        app.input = field.GetComponent<InputField>();
        app.input.textComponent = value;
        app.input.lineType = InputField.LineType.MultiLineNewline;
        app.input.text = "How should Unity handle network requests?";
        Button("Send",canvasGo.transform,new Vector2(30,-270),app.Send);
        Button("Speak",canvasGo.transform,new Vector2(210,-270),app.Speak);
        app.confirmButton=Button("Confirm action",canvasGo.transform,new Vector2(390,-270),app.Confirm);
        Button("Propose rotate",canvasGo.transform,new Vector2(570,-270),app.ProposeRotate);
        app.output = Label("Output",canvasGo.transform,"Start the FastAPI server, then send a question.",new Vector2(30,-340),new Vector2(690,330),19);
        app.status = Label("Status",canvasGo.transform,"Ready",new Vector2(30,-730),new Vector2(1150,45),16);
        Directory.CreateDirectory("Assets/Scenes");
        EditorSceneManager.SaveScene(SceneManager.GetActiveScene(), "Assets/Scenes/Portfolio.unity");
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene("Assets/Scenes/Portfolio.unity",true) };
        PlayerSettings.companyName = "AI XR Portfolio";
        PlayerSettings.productName = "AI XR Portfolio";
        PlayerSettings.defaultScreenWidth = 1280; PlayerSettings.defaultScreenHeight = 800;
        AssetDatabase.SaveAssets();
        Debug.Log("AI_XR_SCENE_GENERATED");
    }

    public static void BuildWindows()
    {
        GenerateScene();
        var result = BuildPipeline.BuildPlayer(new BuildPlayerOptions {
            scenes = new[] { "Assets/Scenes/Portfolio.unity" },
            locationPathName = "Builds/Windows/AI-XR-Portfolio.exe",
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.Development });
        if (result.summary.result != BuildResult.Succeeded)
            throw new Exception("Unity build failed: " + result.summary.result);
        Debug.Log("AI_XR_BUILD_SUCCEEDED " + result.summary.totalSize);
    }
}
