using DeepSharp.RL.Environs;
using DeepSharp.RL.ExperienceSources;

namespace DeepSharp.RL.ExpReplays
{
    /// <summary>
    ///     Exp Relay apply for Store Episode
    /// </summary>
    public class EpisodeExpReplay
    {
        public EpisodeExpReplay(int capacity, float gamma)
        {
            Capacity = capacity;
            Gamma = gamma;
            Buffers = new Queue<Episode>(capacity);
        }

        /// <summary>
        ///     Capacity of Experience Replay Buffer
        /// </summary>
        public int Capacity { protected set; get; }

        /// <summary>
        ///     Capacity of Experience Replay Buffer
        /// </summary>
        public float Gamma { protected set; get; }

        /// <summary>
        ///     Cache
        /// </summary>
        public Queue<Episode> Buffers { set; get; }

        public int Size => Buffers.Sum(a => a.Length);

        public void Enqueue(Episode episode, bool isU = true)
        {
            if (Buffers.Count == Capacity) Buffers.Dequeue();
            if (isU)
            {
                var e = episode.GetReturnEpisode(Gamma);
                Buffers.Enqueue(e);
            }
            else
            {
                Buffers.Enqueue(episode);
            }
        }

        public virtual ExperienceCase All()
        {
            var episodes = Buffers;
            var batchStep = episodes.SelectMany(a => a.Steps).ToArray();

            /// Get Array from Steps
            var stateArray = batchStep.Select(a => a.PreState.Value!.unsqueeze(0)).ToArray();
            var actArray = batchStep.Select(a => a.Action.Value!.unsqueeze(0)).ToArray();
            var rewardArray = batchStep.Select(a => a.Reward.Value).ToArray();
            var stateNextArray = batchStep.Select(a => a.PostState.Value!.unsqueeze(0)).ToArray();
            var doneArray = batchStep.Select(a => a.IsComplete).ToArray();

            /// Convert to VStack
            var state = torch.vstack(stateArray);
            var actionV = torch.vstack(actArray).to(torch.ScalarType.Int64);
            var reward = torch.from_array(rewardArray).view(-1, 1);
            var stateNext = torch.vstack(stateNextArray);
            var done = torch.from_array(doneArray).reshape(Size);

            var excase = new ExperienceCase(state, actionV, reward, stateNext, done);
            return excase;
        }

        public void Clear()
        {
            Buffers.Clear();
        }
    }
}