using System.Collections;
using UnityEngine;
using System.Linq;

public class NeuralNetworkTrainingExample : NeuralNetwork
{
    private IEnumerator m_trainCoroutine;
    private Rigidbody[] m_rigidbodies;
    private float m_trainingStartTime;
    private NeuralNetwork[] m_networks;
    private NeuralNetwork m_bestNeuralNetworkEver;
    private int m_epoch;
    private string m_weightsFileDirectory;

    public Transform target;
    public Transform startPosition;
    public Material gpuInstancedMaterial;

    public delegate void OnNetworksInitializedDelegate();
    public event OnNetworksInitializedDelegate OnNetworksInitializedEvent;

    public delegate void OnNewEpochDelegate();
    public event OnNewEpochDelegate OnNewEpochEvent;

    public Transform canvas;

    [Header("Neural Network Settings")]
    public int[] Layers;
    public int numberOfInputs = 6;
    public int numberOfOutputs = 3;

    [Header("Training Settings")]
    public int numberOfNetworks = 1;
    public int epochTimeInSeconds = 15;
    public float radius = 20f;
    public bool isTraining = true;
    public virtual void Start()
    {
        canvas.gameObject.SetActive(false);
        m_weightsFileDirectory = Application.dataPath + "/weights";
        if (InputsAreValid())
        {
            m_bestNeuralNetworkEver = new NeuralNetwork();
            m_bestNeuralNetworkEver.InitializeNeuralnetwork(Layers, numberOfInputs, numberOfOutputs);

            m_networks = new NeuralNetwork[numberOfNetworks];
            m_rigidbodies = new Rigidbody[numberOfNetworks];

            for (int i = 0; i < numberOfNetworks; i++)
            {
                GameObject obj = GameObject.CreatePrimitive(PrimitiveType.Cube);
                MeshRenderer mr = obj.GetComponent<MeshRenderer>();
                mr.material = gpuInstancedMaterial;
                mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                mr.receiveShadows = false;
                mr.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;
                obj.transform.parent = transform;
                obj.name = "Neural Network #" + i.ToString();
                obj.layer = LayerMask.NameToLayer("Agent");

                m_networks[i] = obj.AddComponent<NeuralNetwork>();
                m_networks[i].InitializeNeuralnetwork(Layers, numberOfInputs, numberOfOutputs);
                m_rigidbodies[i] = obj.AddComponent<Rigidbody>();
            }
            m_bestNeuralNetworkEver.CopyWeights(m_networks[0]);
            m_trainCoroutine = Train();
            //target.transform.position = RandomCircle(startPosition.position, radius);
            Debug.Log("Starting Training");
            m_trainingStartTime = Time.time;
            StartCoroutine(m_trainCoroutine);
        }
    }

    private bool InputsAreValid()
    {
        if (numberOfNetworks < 1)
        {
            Debug.LogError("Number of networks cannot be less than 1");
            return false;
        }
        if (numberOfInputs == 0)
        {
            Debug.LogError("Number of networks inputs cannot be 0");
            return false;
        }
        if (numberOfOutputs == 0)
        {
            Debug.LogError("Number of networks outputs cannot be 0");
            return false;
        }
        return true;
    }

    protected IEnumerator Train()
    {
        while (true)
        {
            if (Input.GetKeyDown(KeyCode.L))
            {
                m_bestNeuralNetworkEver.Load(m_weightsFileDirectory);
                Debug.Log("Loaded best net weights");
            }

            if (Time.time >= m_trainingStartTime + epochTimeInSeconds)
            {
                NewEpoch();
                m_trainingStartTime = Time.time;
                m_bestNeuralNetworkEver.Save(m_weightsFileDirectory);
                yield return null;
            }
            else
            {

                for (int i = 0; i < numberOfNetworks-1; i++)
                {
                    float[] newInputs = new float[6];
                    UpdateInputs(i, newInputs);
                    float[] netOutput = m_networks[i].FeedForward(newInputs);
                    m_rigidbodies[i].AddTorque(new Vector3(netOutput[0], netOutput[1], netOutput[2]));

                }
            }
            yield return null;
        }
    }

    private void NewEpoch()
    {
        m_epoch++;
        Debug.Log("New Epoch: " + m_epoch);
        m_bestNeuralNetworkEver.CopyWeights(GetBestNeuralNetworkInLastEpoch());
        for (int i = 0; i < numberOfNetworks; i++)
        {
            m_networks[i].CopyWeights(m_bestNeuralNetworkEver);
            if (isTraining)
            {
                m_networks[i].Mutate();
            }
            m_networks[i].transform.position = startPosition.position;
            m_networks[i].transform.rotation = startPosition.rotation;
        }
        //target.transform.position = RandomCircle(startPosition.position, radius);
        if(OnNewEpochEvent!=null)
            OnNewEpochEvent();
    }

    private NeuralNetwork GetBestNeuralNetworkInLastEpoch()
    {
        float bestFitness = float.MinValue;
        int bestFitnessIndex = 0;
        for (int i = 0; i < numberOfNetworks; i++)
        {
            m_networks[i].fitness = -Vector3.Distance(m_networks[i].transform.position, target.position);
            if (m_networks[i].fitness > bestFitness)
            {
                bestFitness = m_networks[i].fitness;
                bestFitnessIndex = i;
            }
        }
        return m_networks[bestFitnessIndex];
    }

    private void UpdateInputs(int i, float[] newInputs)
    {
        newInputs[0] = m_networks[i].transform.position.x;
        newInputs[1] = m_networks[i].transform.position.y;
        newInputs[2] = m_networks[i].transform.position.z;
        newInputs[3] = target.transform.position.x;
        newInputs[4] = target.transform.position.y;
        newInputs[5] = target.transform.position.z;
    }

    Vector3 RandomCircle(Vector3 center, float radius)
    {
        float angle = Random.Range(0, Mathf.PI * 2);    // Random angle in radians
        Vector3 newPos = center + new Vector3(Mathf.Sin(angle) * radius, 1f, Mathf.Cos(angle) * radius);
        return newPos;
    }
}
