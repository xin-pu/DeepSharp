using DeepSharp.RL.Models;
using DeepSharp.RL.Policies;
using Action = DeepSharp.RL.Models.Action;

namespace TorchSharpTest.RLTest
{
    /// <summary>
    ///     多臂赌博机,每个赌博机以 0,0.1到1 的概率随机生成
    /// </summary>
    public class KArmedBandit : Envir
    {
        public KArmedBandit(int k)
        {
            K = k;
            var random = new Random();
            bandits = new Bandit[k];
            foreach (var i in Enumerable.Range(0, k))
                bandits[i] = new Bandit($"{i}", random.Next(2, 8) * 1f / 10);

            bandits[0].Prob = 0.3;
            bandits[1].Prob = 0.7;
            Observation = new Observation {Value = torch.zeros(k)};
            Reward = new Reward(0);
        }

        protected int K { set; get; }
        protected Bandit[] bandits { set; get; }

        public override void ResetObservation()
        {
            Observation = new Observation {Value = torch.zeros(K)};
            Reward = new Reward(0);
        }

        /// <summary>
        /// </summary>
        /// <param name="action">动作，该环境下包含智能体选择的赌博机索引</param>
        /// <returns>返回选择的赌博机当次执行后获得的金币数量 0 或 1</returns>
        public override Observation UpdateState(Action action)
        {
            var banditSelectIndex = action.Value.item<long>();
            var bandit = bandits[banditSelectIndex];
            var value = bandit.Step();
            var stateArray = Enumerable.Repeat(0, K).Select(a => (float) a).ToArray();
            stateArray[banditSelectIndex] = value;
            var stateTensor = torch.from_array(stateArray, torch.ScalarType.Float32);
            return new Observation {Value = stateTensor};
        }

        /// <summary>
        ///     该环境下 当次奖励为赌博机的获得金币数量，无需转换
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public override Reward GetReward(Observation observation)
        {
            var sum = observation.Value.sum().item<float>();
            return new Reward(sum);
        }


        public override string ToString()
        {
            var final = string.Join("\r\n", bandits.Select(a => a.ToString()));
            return final;
        }


        /// <summary>
        ///     返回包含20步的一个片段
        /// </summary>
        /// <param name="policy"></param>
        /// <returns></returns>
        public Episode GetEpisode(IPolicy policy)
        {
            var episode = new Episode();
            foreach (var i in Enumerable.Range(0, 20))
            {
                var action = policy.PredictAction(Observation);
                var obs = Observation = UpdateState(action);
                var reward = GetReward(obs);
                episode.Oars.Add(new OAR {Action = action, Observation = obs, Reward = reward});
            }

            episode.SumReward = new Reward(episode.Oars.Sum(a => a.Reward.Value));
            return episode;
        }


        public Episode[] GetBatchs(IPolicy policy, int batchSize = 20)
        {
            ResetObservation();
            var combines = new List<Episode>();
            foreach (var i in Enumerable.Repeat(0, batchSize))
            {
                var episode = GetEpisode(policy);
                combines.Add(episode);
            }

            return combines.ToArray();
        }
    }


    public class Episode
    {
        public Episode()
        {
            Oars = new List<OAR>();
            SumReward = new Reward(0);
        }

        public List<OAR> Oars { set; get; }
        public Reward SumReward { set; get; }
    }


    public struct OAR
    {
        /// <summary>
        ///     动作
        /// </summary>
        public Action Action { set; get; }

        /// <summary>
        ///     动作后的观察
        /// </summary>
        public Observation Observation { set; get; }

        /// <summary>
        ///     动作后的奖励
        /// </summary>
        public Reward Reward { set; get; }
    }
}