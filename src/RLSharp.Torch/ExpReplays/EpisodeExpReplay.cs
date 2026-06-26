using RLSharp.Torch.Environs;
using RLSharp.Torch.ExperienceSources;

namespace RLSharp.Torch.ExpReplays
{
	/// <summary>
	///     Experience replay for storing episodes.
	/// </summary>
	public class EpisodeExpReplay
	{
		public EpisodeExpReplay(int capacity, float gamma)
		{
			Capacity = capacity;
			Gamma    = gamma;
			Buffers  = new Queue<Episode>(capacity);
		}

		/// <summary>
		///     Capacity of experience replay buffer.
		/// </summary>
		public int Capacity { get; protected set; }

		/// <summary>
		///     Discount factor.
		/// </summary>
		public float Gamma { get; protected set; }

		/// <summary>
		///     Internal buffer queue.
		/// </summary>
		public Queue<Episode> Buffers { get; set; }

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
			var episodes  = Buffers;
			var batchStep = episodes.SelectMany(a => a.Steps).ToArray();

			// Get arrays from steps
			var stateArray     = batchStep.Select(a => a.PreState.Value!.unsqueeze(0)).ToArray();
			var actArray       = batchStep.Select(a => a.Action.Value!.unsqueeze(0)).ToArray();
			var rewardArray    = batchStep.Select(a => a.Reward.Value).ToArray();
			var stateNextArray = batchStep.Select(a => a.PostState.Value!.unsqueeze(0)).ToArray();
			var doneArray      = batchStep.Select(a => a.IsComplete).ToArray();

			// Convert to vstack tensors
			var state     = torch.vstack(stateArray);
			var actionV   = torch.vstack(actArray).to(torch.ScalarType.Int64);
			var reward    = torch.from_array(rewardArray).view(-1, 1);
			var stateNext = torch.vstack(stateNextArray);
			var done      = torch.from_array(doneArray).reshape(Size);

			var excase = new ExperienceCase(state, actionV, reward, stateNext, done);
			return excase;
		}

		public void Clear()
		{
			Buffers.Clear();
		}
	}
}