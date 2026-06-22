using DeepSharp.RL.Environs;
using DeepSharp.RL.ExpReplays;

namespace DeepSharp.RL.Agents.Deep.Value;

/// <summary>
///     Noisy DQN.
///     Uses NoisyNet for parameterized exploration instead of ε-greedy.
///     Noise parameters automatically tune exploration during training — no manual ε decay needed.
///     Noise is reset at the start of each Learn(); each episode uses the same noise.
/// </summary>
public class NoisyDQN : DeepValueAgent
{
    public NoisyDQN(Environ<Space, Space> env,
        int   n         = 1000,
        int   c         = 10000,
        float gamma     = 0.99f,
        int   batchSize = 32)
        : base(env, "NoisyDQN")
    {
        C         = c;
        N         = n;
        BatchSize = batchSize;
        Gamma     = gamma;
        Epsilon   = 0f; // NoisyDQN does not use ε-greedy

        Q       = new NoisyNet(ObservationSize, 128, ActionSize, DeviceType.CPU);
        QTarget = new NoisyNet(ObservationSize, 128, ActionSize, DeviceType.CPU);
        QTarget.load_state_dict(Q.state_dict());

        Optimizer  = SGD(Q.parameters(), 0.001);
        Loss       = MSELoss();
        UniformExp = new UniformExpReplay(C);
    }

    public float Gamma { get; }

    public int C { get; }

    public int N { get; }

    public int BatchSize { get; }

    public Module<torch.Tensor, torch.Tensor> QTarget { get; protected set; }

    public UniformExpReplay UniformExp { get; }

    /// <summary>
    ///     Get inner NoisyNet references (for ResetNoise).
    /// </summary>
    private NoisyNet QNoisy => (NoisyNet)Q;
    private NoisyNet QTargetNoisy => (NoisyNet)QTarget;

    /// <summary>
    ///     Learning loop: reset noise before each episode, no ε-greedy.
    /// </summary>
    public override LearnOutcome Learn()
    {
        var learnOutCome = new LearnOutcome();

        foreach (var _ in Enumerable.Range(0, N))
        {
            // Reset noise before each episode
            QNoisy.ResetNoise();

            Environ.Reset();
            var epoch   = 0;
            var episode = new Episode();
            while (!Environ.IsComplete(epoch))
            {
                epoch++;
                // NoisyNet: use argmax directly (noise provides exploration)
                var act  = GetPolicyAct(Environ.Observation!.Value!);
                var step = Environ.Step(act, epoch);
                episode.Enqueue(step);

                Environ.CallBack?.Invoke(step);
                Environ.Observation = step.PostState;
            }

            learnOutCome.AppendStep(episode);
            UniformExp.Enqueue(episode);
            if (UniformExp.Buffers.Count >= C)
                learnOutCome.Evaluate = UpdateNet();
        }

        SyncTargetNetwork();
        return learnOutCome;
    }

    private float UpdateNet()
    {
        QNoisy.ResetNoise();
        QTargetNoisy.ResetNoise();

        var batchSample = UniformExp.Sample(BatchSize);

        var stateActionValue = Q.forward(batchSample.PreState)
            .gather(1, batchSample.Action).squeeze(-1);

        var nextStateValue            = QTarget.forward(batchSample.PostState).max(1).values.detach();
        var expectedStateActionValue = batchSample.Reward + Gamma * nextStateValue;

        var loss = Loss.call(stateActionValue, expectedStateActionValue);

        Optimizer.zero_grad();
        loss.backward();
        Optimizer.step();
        return loss.item<float>();
    }

    private void SyncTargetNetwork()
    {
        var parameters = Q.state_dict();
        QTarget.load_state_dict(parameters);
    }
}
