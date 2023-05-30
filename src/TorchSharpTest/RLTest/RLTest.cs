using DeepSharp.RL.Models;
using Action = DeepSharp.RL.Models.Action;

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
        public void Main()
        {
            var k = 2;
            var batchSize = 1000;
            var random = new Random();

            /// Step 1 创建环境
            var kArmedBandit = new KArmedBandit(k);
            kArmedBandit.Reset();
            Print(kArmedBandit);

            /// Step 2 创建智能体
            var agent = new AgentKArmedBandit(k, k);

            var actions = new List<Action>();
            var rewards = new List<Reward>();

            foreach (var i in Enumerable.Range(0, 1000))
            {
                var reward = kArmedBandit.Reward;
                rewards.Add(reward);

                var action = agent.PredictAction(reward);
                actions.Add(action);
                Print($"{i}\t{action}");
                agent.Learn(rewards.Select(a => a.Value).ToList(), actions.Select(a => a.Value).ToList());
            }
        }
    }
}