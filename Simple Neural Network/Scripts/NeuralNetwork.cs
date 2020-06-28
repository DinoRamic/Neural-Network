using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;

/// <summary>
/// Simple feedforward neural network with a genetic algorithm
/// </summary>
public class NeuralNetwork : MonoBehaviour
{
    protected int[] m_layers;
    protected float[][] m_neurons;  //[Layer][NeuronValue]
    protected float[][] m_weights; //[Layer][WeightValue
    [HideInInspector] public float fitness = float.MinValue;
    [HideInInspector] public float[] outputs;

    /// <summary>
    /// Initializes a new network with random weights
    /// </summary>
    public void InitializeNeuralnetwork(int[] Layers, int nrOfInputs, int nrOfOutputs)
    {
        m_layers = Layers;
        outputs = new float[nrOfOutputs];
        m_weights = new float[m_layers.Length - 1][];
        fitness = float.MinValue;

        //Initialize weights
        for (int i = 0; i < m_layers.Length - 1; i++)
        {
            m_weights[i] = new float[m_layers[i] * m_layers[i + 1]];
        }

        //Initialize Neurons [neuron][value]
        m_neurons = new float[m_layers.Length][];
        for (int i = 0; i < m_layers.Length; i++)
        {
            m_neurons[i] = new float[m_layers[i]];
        }

        //Assign random weights
        for (int layer = 0; layer < m_layers.Length - 1; layer++)
        {
            for (int neuron = 0; neuron < m_layers[layer]; neuron++)
            {
                m_weights[layer][neuron] = UnityEngine.Random.Range(-1f, 1f);
            }
        }
    }

    /// <summary>
    /// Copies the weights of a network.
    /// </summary>
    public void CopyWeights(NeuralNetwork network)
    {
        DeepCopy(network.m_weights);
    }
    public float[] FeedForward(float[] inputs)
    {
        for (int i = 0; i < m_layers[0]; i++)
        {
            m_neurons[0][i] = inputs[i];
        }

        float weightedSum = 0;
        for (int i = 0; i < m_layers.Length - 1; i++)
        {
            for (int s = 0; s < m_layers[i + 1]; s++)
            {
                weightedSum = 0;
                for (int d = 0; d < m_layers[i]; d++)
                {
                    weightedSum += m_neurons[i][d] * m_weights[i][d];
                }
                m_neurons[i + 1][s] = ActivationFunction(weightedSum);
            }
        }

        outputs = m_neurons[m_layers.Length - 1];
        return outputs;
    }

    /// <summary>
    /// Default activation function is Leaky Relu. Override this function to change it.
    /// </summary>
    protected virtual float ActivationFunction(float weightedAverageSum)
    {
        if (weightedAverageSum > 0)
        {
            return weightedAverageSum;
        }
        else
        {
            return 0.01f * weightedAverageSum;
        }
    }


    /// <summary>
    /// Changes the current weights of the neural network based on a random number.
    /// </summary>
    public virtual void Mutate()
    {
        for (int i = 0; i < m_layers.Length - 1; i++)
        {
            for (int d = 0; d < m_layers[i]; d++)
            {
                int randomNumber = UnityEngine.Random.Range(0, 100);
                if (randomNumber < 30)
                {
                    m_weights[i][d] = m_weights[i][d] + UnityEngine.Random.Range(-4f, 4f);
                }
                else if (randomNumber < 60)
                {
                    m_weights[i][d] = m_weights[i][d] * UnityEngine.Random.Range(-4f, 4f);
                }
                else if (randomNumber < 90)
                {
                    m_weights[i][d] = -m_weights[i][d];
                }
            }
        }
    }
    /// <summary>
    /// Makes a deep copy of the weights.
    /// </summary>
    private void DeepCopy(float[][] weightsToCopy)
    {
        for (int i = 0; i < m_weights.Length; i++)
        {
            m_weights[i] = (float[])weightsToCopy[i].Clone();
        }
    }

    /// <summary>
    /// Load weights from a file and save them to this neural network.
    /// </summary>
    public void Load(string filepath)
    {
        using (BinaryReader reader = new BinaryReader(File.Open(filepath, FileMode.OpenOrCreate)))
        {
            for (int i = 0; i < m_weights.Length; i++)
            {
                for (int d = 0; d < m_weights[i].Length; d++)
                {
                    byte[] bytes = reader.ReadBytes(4);
                    m_weights[i][d] = System.BitConverter.ToSingle(bytes, 0);
                }
            }
        }
    }

    /// <summary>
    /// Save the weights from this neural network to a file.
    /// </summary>
    public void Save(string filepath)
    {
        using (BinaryWriter writer = new BinaryWriter(File.Open(filepath, FileMode.OpenOrCreate)))
        {
            for (int i = 0; i < m_weights.Length; i++)
            {
                for(int d = 0; d < m_weights[i].Length; d++)
                {
                    writer.Write(m_weights[i][d]);
                }
            }
        }
    }
}
