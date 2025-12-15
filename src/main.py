"""Main entry point for Chess Vision Engine."""

from __future__ import annotations

from pathlib import Path

import click

from src.detection import BoardDetector, PieceClassifier
from src.chess_logic import FENGenerator
from src.chess_logic.position_validator import PositionValidator
from src.engine import StockfishWrapper
from src.utils.config import settings
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(verbose: bool) -> None:
    """Chess Vision Engine - Scan chessboards and analyze positions."""
    log_level = "DEBUG" if verbose else settings.log_level
    setup_logging(log_level)


@cli.command()
@click.option(
    "--image",
    "-i",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to chessboard image",
)
@click.option("--depth", "-d", default=20, show_default=True, help="Engine analysis depth")
@click.option("--output", "-o", type=click.Path(dir_okay=False), help="Output file for analysis")
def analyze(image: str, depth: int, output: str | None) -> None:
    """Analyze a chessboard image."""
    logger.info("Analyzing image: %s", image)

    detector = BoardDetector()
    classifier = PieceClassifier()
    fen_generator = FENGenerator()
    validator = PositionValidator()

    try:
        logger.info("Detecting board...")
        board_image = detector.detect_board(image)

        logger.info("Classifying pieces...")
        pieces = classifier.classify_pieces(board_image)

        logger.info("Generating FEN notation...")
        fen = fen_generator.generate(pieces)
        click.echo(f"Detected position: {fen}")

        # Validate BEFORE calling the engine
        validation = validator.validate(fen)
        if not validation.is_valid:
            click.echo("❌ Invalid position detected; skipping engine analysis.")
            for e in validation.errors:
                click.echo(f"  - {e}")
            for w in validation.warnings:
                click.echo(f"  (warn) {w}")

            suggestions = validator.suggest_corrections(fen)
            if suggestions:
                # de-dupe suggestions while preserving order
                deduped = list(dict.fromkeys(suggestions))
                click.echo("Suggestions:")
                for s in deduped:
                    click.echo(f"  * {s}")

            raise click.ClickException("Detected position is invalid.")

        logger.info("Running engine analysis (depth %d)...", depth)

        # IMPORTANT: Always close Stockfish so the CLI exits cleanly
        with StockfishWrapper() as engine:
            analysis = engine.analyze(fen, depth=depth)

        click.echo("\nAnalysis Results:")
        click.echo(f"  Best move: {analysis.best_move}")
        click.echo(f"  Evaluation: {analysis.score}")
        click.echo(f"  Principal variation: {' '.join(analysis.pv[:5])}")

        if output:
            _save_analysis(output, fen, analysis)
            click.echo(f"\nResults saved to: {output}")

    except click.ClickException:
        raise
    except Exception as e:
        logger.exception("Analysis failed")
        raise click.ClickException(str(e))


@cli.command()
@click.option("--camera", "-c", default=0, show_default=True, help="Camera index")
@click.option("--depth", "-d", default=15, show_default=True, help="Engine analysis depth")
def live(camera: int, depth: int) -> None:
    """Start live webcam analysis."""
    logger.info("Starting live analysis with camera %d", camera)

    try:
        import cv2
    except ImportError as e:
        raise click.ClickException("Live analysis requires OpenCV. Install with: pip install opencv-python") from e

    detector = BoardDetector()
    classifier = PieceClassifier()
    fen_generator = FENGenerator()
    validator = PositionValidator()

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise click.ClickException(f"Could not open camera index {camera}")

    click.echo("Press 'q' to quit, 'a' to analyze current frame")

    # Keep Stockfish alive during the session; auto-close on exit
    try:
        with StockfishWrapper() as engine:
            while True:
                ret, frame = cap.read()
                if not ret:
                    click.echo("Failed to read from camera.")
                    break

                cv2.imshow("Chess Vision Engine", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if key == ord("a"):
                    try:
                        board = detector.detect_board_from_array(frame)
                        pieces = classifier.classify_pieces(board)
                        fen = fen_generator.generate(pieces)

                        click.echo(f"\nFEN: {fen}")

                        validation = validator.validate(fen)
                        if not validation.is_valid:
                            click.echo("❌ Invalid position; skipping engine analysis.")
                            for e in validation.errors:
                                click.echo(f"  - {e}")
                            continue

                        analysis = engine.analyze(fen, depth=depth)
                        click.echo(f"Best move: {analysis.best_move}")
                        click.echo(f"Evaluation: {analysis.score}")

                    except Exception as e:
                        click.echo(f"Analysis failed: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


@cli.command()
@click.option("--port", "-p", default=8080, show_default=True, help="Server port")
@click.option("--host", "-h", default="0.0.0.0", show_default=True, help="Server host")
def serve(port: int, host: str) -> None:
    """Start the web server."""
    import uvicorn
    from src.ui.app import app

    logger.info("Starting server at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.argument("fen")
@click.option("--depth", "-d", default=20, show_default=True, help="Engine analysis depth")
def engine(fen: str, depth: int) -> None:
    """Analyze a FEN position with the engine."""
    try:
        with StockfishWrapper() as engine_obj:
            result = engine_obj.analyze(fen, depth=depth)

        click.echo(f"Position: {fen}")
        click.echo(f"Best move: {result.best_move}")
        click.echo(f"Evaluation: {result.score}")
        click.echo(f"Depth: {result.depth}")
        click.echo(f"PV: {' '.join(result.pv[:10])}")

    except Exception as e:
        raise click.ClickException(str(e))


def _save_analysis(path: str, fen: str, analysis) -> None:
    """Save analysis results to file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"FEN: {fen}\n")
        f.write(f"Best Move: {analysis.best_move}\n")
        f.write(f"Evaluation: {analysis.score}\n")
        f.write(f"PV: {' '.join(analysis.pv)}\n")


if __name__ == "__main__":
    cli()
