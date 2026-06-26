using RLSharp.Torch.Environs;
using RLSharp.Torch.ExperienceSources;

namespace RLSharp.Torch.ExpReplays
{
	/// <summary>
	///     Experience replay for storing steps [State, Action, Reward, NextState].
	/// </summary>
	public abstract class ReplayBuffer
	{
		protected ReplayBuffer(int capacity = 10000)
		{
			Capacity = capacity;
			if (capacity <= 0)
				throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be positive.");

			Buffers = new Queue<Step>(capacity);
		}

		/// <summary>
		///     Capacity of experience replay buffer.
		/// </summary>
		public int Capacity { get; protected set; }

		/// <summary>
		///     Internal buffer queue.
		/// </summary>
		public Queue<Step> Buffers { get; }

		public int Size => Buffers.Count();

		/// <summary>
		///     Record one step [State, Action, Reward, NewState].
		/// </summary>
		/// <param name="step"></param>
		public virtual void Enqueue(Step step)
		{
			if (Buffers.Count == Capacity) Buffers.Dequeue();
			Buffers.Enqueue(step);
		}

		/// <summary>
		///     Record multiple steps.
		/// </summary>
		public void Enqueue(IEnumerable<Step> steps)
		{
			steps.ToList().ForEach(Enqueue);
		}

		protected abstract Step[] SampleSteps(int batchsize);

		public virtual void Enqueue(Episode episode)
		{
			Enqueue(episode.Steps);
		}

		public virtual ExperienceCase Sample(int batchsize)
		{
			if (batchsize <= 0)
				throw new ArgumentOutOfRangeException(nameof(batchsize), "Batch size must be positive.");
			if (Buffers.Count == 0)
				throw new InvalidOperationException("Cannot sample from an empty replay buffer.");

			var batchStep = SampleSteps(batchsize);

			// Get arrays from steps
			var stateArray     = batchStep.Select(a => a.PreState.Value!.unsqueeze(0)).ToArray();
			var actArray       = batchStep.Select(a => a.Action.Value!.unsqueeze(0)).ToArray();
			var rewardArray    = batchStep.Select(a => a.Reward.Value).ToArray();
			var stateNextArray = batchStep.Select(a => a.PostState.Value!.unsqueeze(0)).ToArray();
			var doneArray      = batchStep.Select(a => a.IsComplete).ToArray();

			try
			{
				var       state       = torch.vstack(stateArray);
				using var actionStack = torch.vstack(actArray);
				var       actionV     = actionStack.to(torch.ScalarType.Int64);
				var       reward      = torch.from_array(rewardArray);
				var       stateNext   = torch.vstack(stateNextArray);
				var       done        = torch.from_array(doneArray);

				return new ExperienceCase(state, actionV, reward, stateNext, done);
			}
			finally
			{
				foreach (var tensor in stateArray) tensor.Dispose();
				foreach (var tensor in actArray) tensor.Dispose();
				foreach (var tensor in stateNextArray) tensor.Dispose();
			}
		}


		public void Clear()
		{
			Buffers.Clear();
		}
	}
}