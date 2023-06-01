using DeepSharp.RL.Agents;

namespace TorchSharpTest.RLTest
{
    public class RLTest : AbstractTest
    {
        public RLTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void BanditTest()
        {
            var count = 0;
            var bandit = new Bandit("A");
            var range = 100;
            foreach (var i in Enumerable.Range(0, range))
            {
                var res = bandit.Step();
                if (res > 0) count++;
                Print($"{i:D4}:{res}");
            }

            Print(count * 1f / range);
        }

        [Fact]
        public void KArmedBanditTest()
        {
            var kArmedBandit = new KArmedBandit(5);
            Print(kArmedBandit);
        }

        [Fact]
        public void RandomPickup()
        {
            var probs = new[] {0.5f, 0.5f};

            foreach (var i in Enumerable.Repeat(0, 10))
            {
                var res = torch.multinomial(torch.from_array(probs), 1);
                var index = res.item<long>();
                Print(index);
            }
        }

        [Fact]
        public void Main()
        {
            var k = 4;
            var batchSize = 1000;
            var percent = 0.7f;

            /// Step 1 创建环境
            var kArmedBandit = new KArmedBandit(k);
            Print(kArmedBandit);

            /// Step 2 创建智能体
            var agent = new AgentCrossEntropy(k, k);

            /// Step 3 边收集 边学习
            foreach (var i in Enumerable.Range(0, 200))
            {
                var batch = kArmedBandit.GetMultiEpisodes(agent, 20);
                var oars = agent.GetElite(batch, percent);

                var observation = torch.vstack(oars.Select(a => a.Observation.Value).ToList());
                var action = torch.vstack(oars.Select(a => a.Action.Value).ToList());

                var rewardMean = batch.Select(a => a.SumReward.Value).Average();
                var loss = agent.Learn(observation, action);

                Print($"Epoch:{i:D4}\tReward:{rewardMean:F4}\tLoss:{loss:F4}");
            }
        }
    }
}