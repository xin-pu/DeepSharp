using DeepSharp.RL.Environs;
using FluentAssertions;
using static TorchSharp.torch.optim;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     An Agent base on CrossEntropy Function
    ///     Cross-Entropy Method
    ///     http://people.smp.uq.edu.au/DirkKroese/ps/eormsCE.pdf
    /// </summary>
    public class AgentCrossEntropy : Agent
    {
        public AgentCrossEntropy(Environ<Space, Space> environ,
            float percentElite = 0.7f,
            int hiddenSize = 100) : base(environ)
        {
            PercentElite = percentElite;
            AgentNet = new Net((int) environ.ObservationSpace.N, hiddenSize, (int) environ.ActionSpace.N);
            Optimizer = Adam(AgentNet.parameters(), 0.01);
            Loss = CrossEntropyLoss();
        }


        public float PercentElite { protected set; get; }

        public Net AgentNet { protected set; get; }

        public Optimizer Optimizer { protected set; get; }

        public Loss<torch.Tensor, torch.Tensor, torch.Tensor> Loss { protected set; get; }

        /// <summary>
        ///     智能体 根据观察 生成动作 概率 分布，并按分布生成下一个动作
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public override Act PredictAction(Observation observation)
        {
            var input = observation.Value!.unsqueeze(0);
            var sm = Softmax(1);
            var actionProbs = sm.forward(AgentNet.forward(input));
            var nextAction = torch.multinomial(actionProbs, 1);
            return new Act(nextAction);
        }


        public override float Learn(Episode[] steps)
        {
            if (steps.Length == 0) return float.MaxValue;


            var oars = steps.SelectMany(a => a.Oars)
                .ToList();

            var observations = oars
                .Select(a => a.Observation.Value)
                .ToList();
            var actions = oars
                .Select(a => a.Action.Value)
                .ToList();

            var observation = torch.vstack(observations!);
            var action = torch.vstack(actions!);

            return Learn(observation, action);
        }

        /// <summary>
        ///     Replace default Optimizer
        /// </summary>
        /// <param name="optimizer"></param>
        public void UpdateOptimizer(Optimizer optimizer)
        {
            Optimizer = optimizer;
        }

        /// <summary>
        ///     Get Elite
        /// </summary>
        /// <param name="episodes"></param>
        /// <param name="percent"></param>
        /// <returns></returns>
        public virtual Episode[] GetElite(Episode[] episodes)
        {
            var reward = episodes
                .Select(a => a.SumReward.Value)
                .ToArray();
            var rewardP = reward.OrderByDescending(a => a)
                .Take((int) (reward.Length * PercentElite))
                .Min();

            var filterEpisodes = episodes
                .Where(e => e.SumReward.Value > rewardP)
                .ToArray();

            return filterEpisodes;
        }

        /// <summary>
        ///     core function to update net
        /// </summary>
        /// <param name="observations">tensor from multi observations, size: [batch,observation size]</param>
        /// <param name="actions">tensor from multi actions, size: [batch,action size]</param>
        /// <returns>loss</returns>
        public float Learn(torch.Tensor observations, torch.Tensor actions)
        {
            observations.shape.Last().Should()
                .Be(ObservationSize, $"Agent observations tensor should be [B,{ObservationSize}]");

            actions = actions.squeeze(-1);
            var pred = AgentNet.forward(observations);
            var output = Loss.call(pred, actions);

            Optimizer.zero_grad();
            output.backward();
            Optimizer.step();

            var loss = output.item<float>();
            return loss;
        }


        /// <summary>
        ///     This is demo net to guide how to create a new Module
        /// </summary>
        public sealed class Net : Module<torch.Tensor, torch.Tensor>
        {
            private readonly Module<torch.Tensor, torch.Tensor> layers;

            public Net(int obsSize, int hiddenSize, int actionNum) : base("Net")
            {
                var modules = new List<(string, Module<torch.Tensor, torch.Tensor>)>
                {
                    ("line1", Linear(obsSize, hiddenSize)),
                    ("relu", ReLU()),
                    ("line2", Linear(hiddenSize, actionNum))
                };
                layers = Sequential(modules);
                layers.to(new torch.Device(DeviceType.CUDA));
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor input)
            {
                return layers.forward(input.to_type(torch.ScalarType.Float32));
            }

            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    layers.Dispose();
                    ClearModules();
                }

                base.Dispose(disposing);
            }
        }
    }
}