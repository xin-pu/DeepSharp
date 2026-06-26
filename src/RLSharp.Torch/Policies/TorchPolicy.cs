using RLSharp.Core.Agents;
using RLSharp.Torch.Encoding;

namespace RLSharp.Torch.Policies
{
	public sealed class TorchPolicy<TState, TAction> : IPolicy<TState, TAction>
		where TAction : notnull
	{
		public TorchPolicy(
			Module<torch.Tensor, torch.Tensor> network,
			IStateEncoder<TState>              stateEncoder,
			DiscreteActionEncoder<TAction>     actionEncoder)
		{
			Network       = network       ?? throw new ArgumentNullException(nameof(network));
			StateEncoder  = stateEncoder  ?? throw new ArgumentNullException(nameof(stateEncoder));
			ActionEncoder = actionEncoder ?? throw new ArgumentNullException(nameof(actionEncoder));
		}

		public Module<torch.Tensor, torch.Tensor> Network { get; }

		public IStateEncoder<TState> StateEncoder { get; }

		public DiscreteActionEncoder<TAction> ActionEncoder { get; }

		public TAction SelectAction(TState state)
		{
			using var input       = StateEncoder.Encode(state).unsqueeze(0);
			using var output      = Network.forward(input).squeeze(0);
			var       actionIndex = output.argmax().ToInt64();
			return ActionEncoder.FromIndex(actionIndex);
		}

		public void Save(string path)
		{
			Network.save(path);
		}

		public static TorchPolicy<TState, TAction> Load(
			string                             path,
			Module<torch.Tensor, torch.Tensor> network,
			IStateEncoder<TState>              stateEncoder,
			DiscreteActionEncoder<TAction>     actionEncoder)
		{
			network.load(path);
			return new TorchPolicy<TState, TAction>(network, stateEncoder, actionEncoder);
		}
	}
}