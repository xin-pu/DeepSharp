using DeepSharp.RL.Environs;
using DeepSharp.RL.ExpReplays;

namespace DeepSharp.RL.Agents.Deep.Value;

/// <summary>
///     Dueling DQN
///     Uses DuelingNet to decompose Q(s,a) into state value V(s) + advantage A(s,a),
///     learning state values more stably, especially in environments where actions have small impact on value.
/// </summary>
public class DuelingDQN : DeepValueAgent
{
    public DuelingDQN(Environ<Space, Space> env,
        int   n         = 1000,
        int   c         = 10000,
        float epsilon   = 0.1f,
        float gamma     = 0.99f,
        int   batchSize = 32)
        : base(env, "DuelingDQN")
    {
        C         = c;
        N         = n;
        BatchSize = batchSize;
        Epsilon   = epsilon;
        Gamma     = gamma;

        Q       = new DuelingNet(ObservationSize, 128, ActionSize, DeviceType.CPU);
        QTarget = new DuelingNet(ObservationSize, 128, ActionSize, DeviceType.CPU);
        QTarget.load_state_dict(Q.state_dict());

        Optimizer  = SGD(Q.parameters(), 0.001);
        Loss       = MSELoss();
        UniformExp = new UniformExpReplay(C);
    }

    public float Gamma { get; }

    /// <summary>
    ///     Experience replay capacity.
    /// </summary>
    public int C { get; }

    /// <summary>
    ///     Update interval.
    /// </summary>
    public int N { get; }

    /// <summary>
    ///     Batch size.
    /// </summary>
    public int BatchSize { get; }

    /// <summary>
    ///     Target network (also uses Dueling architecture).
    /// </summary>
    public Module<torch.Tensor, torch.Tensor> QTarget { get; protected set; }

    /// <summary>
    ///     Experience replay buffer.
    /// </summary>
    public UniformExpReplay UniformExp { get; }

    /// <summary>
    ///     Learning loop (same as DQN, differs only in the DuelingNet architecture).
    /// </summary>
    public override LearnOutcome Learn()
    {
        var learnOutCome = new LearnOutcome();
        foreach (var _ in Enumerable.Range(0, N))
        {
            Environ.Reset();
            var epoch   = 0;
            var episode = new Episode();
            while (!Environ.IsComplete(epoch))
            {
                epoch++;
                var act  = GetEpsilonAct(Environ.Observation!.Value!);
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

    /// <summary>
    ///     Standard DQN update (using DuelingNet Q values).
    /// </summary>
    private float UpdateNet()
    {
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
