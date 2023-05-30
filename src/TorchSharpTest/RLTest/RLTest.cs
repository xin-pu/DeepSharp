using DeepSharp.RL.CrossEntropy;

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

            var net = new Net(k, 10, k);

            var actions = new List<torch.Tensor>();
            var rewards = new List<torch.Tensor>();

            foreach (var i in Enumerable.Range(0, 1))
                if (i < batchSize)
                {
                    var index = random.Next(0, 2);
                }
        }
    }
}