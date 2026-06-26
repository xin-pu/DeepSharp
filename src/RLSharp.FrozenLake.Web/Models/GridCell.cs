namespace RLSharp.FrozenLake.Web.Models
{
	public class GridCell
	{
		public int Index { get; set; }

		public int Row { get; set; }

		public int Column { get; set; }

		/// <summary>
		///     Cell role: "Start", "Ice", "Hole", "End".
		/// </summary>
		public string Role { get; set; } = "Ice";

		public bool IsPlayer { get; set; }
	}
}