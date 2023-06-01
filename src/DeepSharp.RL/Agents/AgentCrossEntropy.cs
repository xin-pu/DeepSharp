using DeepSharp.RL.Models;
using DeepSharp.RL.Policies;
using FluentAssertions;
using static TorchSharp.torch.optim;
using Action = DeepSharp.RL.Models.Action;

namespace DeepSharp.RL.Agents
{
    public class AgentCrossEntropy : IPolicy
    {
        public AgentCrossEntropy(int obsSize, int actionSize, int hiddenSize = 100)
        {
            ObservationSize = obsSize;
            ActionSize = actionSize;
            AgentNet = new Net(obsSize, hiddenSize, actionSize);
            Optimizer = Adam(AgentNet.parameters(), 1E-2);
            Loss = CrossEntropyLoss();
        }

        public int ObservationSize { protected set; get; }
        public int ActionSize { protected set; get; }


        public Net AgentNet { protected set; get; }

        public Optimizer Optimizer { protected set; get; }

        public Loss<torch.Tensor, torch.Tensor, torch.Tensor> Loss { protected set; get; }

        /// <summary>
        ///     Replace default Optimizer
        /// </summary>
        /// <param name="optimizer"></param>
        public void UpdateOptimizer(Optimizer optimizer)
        {
            Optimizer = optimizer;
        }

        /// <summary>
        ///     智能体 根据观察 生成动作 概率 分布，并按分布生成下一个动作
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public Action PredictAction(Observation observation)
        {
            var input = observation.Value.unsqueeze(0);
            var sm = Softmax(1);
            var actionProbs = sm.forward(AgentNet.forward(input));
            return new Action(actionProbs);
        }

        /// <summary>
        ///     Get Elite
        /// </summary>
        /// <param name="episodes"></param>
        /// <param name="percent"></param>
        /// <returns></returns>
        public Step[] GetElite(Episode[] episodes, double percent)
        {
            var reward = episodes
                .Select(a => a.SumReward.Value)
                .ToArray();
            var rewardP = reward.OrderByDescending(a => a)
                .Take((int) (reward.Length * percent))
                .Min();

            var elite = episodes
                .Where(e => e.SumReward.Value > rewardP)
                .SelectMany(e => e.Oars)
                .ToArray();

            return elite;
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
            actions.shape.Last().Should()
                .Be(ActionSize, $"Agent actions tensor should be [B,{ActionSize}]");

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
                RegisterComponents();
            }


            public override torch.Tensor forward(torch.Tensor input)
            {
                return layers.forward(input);
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