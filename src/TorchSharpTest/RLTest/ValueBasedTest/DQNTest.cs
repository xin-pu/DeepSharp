using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using FluentAssertions;

namespace TorchSharpTest.RLTest.ValueBasedTest
{
    public class DQNTest : AbstractTest
    {
        public DQNTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        [Fact]
        public void TestDQNModel()
        {
            var deviceType = DeviceType.CPU;
            var scalarType = torch.ScalarType.Float32;

            var net = new DQNNet(new long[] {1, 416, 416}, 3,
                scalarType,
                deviceType);

            var c = net.children();
            foreach (var module in c) Print(module.GetName());

            var input = torch.randn(1, 1, 416, 416, scalarType, new torch.Device(deviceType));
            var res = net.forward(input);
            Print(res);

            res.shape.Should().BeEquivalentTo(new long[] {1, 3});
        }


        [Fact]
        public void TestDQN()
        {
            var frozenLake = new Frozenlake();
            var dqn = new DQN(frozenLake, 10, 10);
            var act = dqn.GetPolicyAct(frozenLake.Observation!.Value!);
            Print(act);
        }


        [Fact]
        public void KArmedBanditMain()
        {
            /// Step 1 Create a 4-Armed Bandit
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.80, 0.72, 0.70}) {Gamma = 0.95f};

            /// Step 2 Create AgentCrossEntropy with 0.7f percentElite as default
            var agent = new DQN(kArmedBandit, 1, 100);
            Print(kArmedBandit);

            var i = 0;
            var testEpisode = 20;
            var bestReward = 0f;
            while (true)
            {
                i++;
                kArmedBandit.Reset();
                agent.Learn();


                var episode = agent.RunEpisodes(testEpisode);
                var reward = episode.Average(a => a.SumReward.Value);

                bestReward = new[] {bestReward, reward}.Max();
                Print($"{agent} Play:{i:D3}\t {reward}");
                if (bestReward > 16.2)
                    break;
            }

            var e = agent.RunEpisode();
            var act = e.Steps.Select(a => a.Action);
            Print(string.Join("\r\n", act));
        }
    }
}