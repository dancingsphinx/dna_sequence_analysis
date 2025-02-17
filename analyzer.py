"""
DNA Analysis Tool
----------------
A comprehensive toolkit for DNA sequence analysis and manipulation.

This module provides a collection of tools for:
- DNA sequence validation and manipulation
- Sequence property analysis (GC content, melting temperature)
- Pattern searching and alignment
- Random sequence generation
- Thermodynamic calculations

References:
-----------
[1] SantaLucia J Jr. (1998) "A unified view of polymer, dumbbell, and oligonucleotide DNA
    nearest-neighbor thermodynamics", PNAS, 95(4):1460-1465.
[2] Smith TF, Waterman MS. (1981) "Identification of common molecular subsequences",
    J Mol Biol, 147(1):195-197.
[3] Breslauer KJ et al. (1986) "Predicting DNA duplex stability from the base sequence",
    PNAS, 83(11):3746-3750.
"""

import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Constants
DNA_BASES = {'A', 'T', 'C', 'G'}
COMPLEMENTARY_BASES = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
R_CONSTANT = 1.987  # cal/(K*mol) - Gas constant


@dataclass
class ThermodynamicParameters:
    """Storage class for nearest-neighbor thermodynamic parameters."""

    # ΔH° values (kcal/mol) for nearest-neighbor pairs
    # Data from SantaLucia (1998)
    DELTA_H = {
        "AA/TT": -7.9, "TT/AA": -7.9, "AT/TA": -7.2, "TA/AT": -7.2,
        "CA/GT": -8.5, "AC/TG": -8.5, "GT/CA": -8.4, "TG/AC": -8.4,
        "CT/GA": -7.8, "TC/AG": -7.8, "AG/TC": -8.2, "GA/CT": -8.2,
        "CG/GC": -10.6, "GC/CG": -9.8, "GG/CC": -8.0, "CC/GG": -8.0
    }

    # ΔS° values (cal/K·mol) for nearest-neighbor pairs
    DELTA_S = {
        "AA/TT": -22.2, "TT/AA": -22.2, "AT/TA": -20.4, "TA/AT": -21.3,
        "CA/GT": -22.7, "AC/TG": -22.7, "GT/CA": -22.4, "TG/AC": -8.4,
        "CT/GA": -21.0, "TC/AG": -21.0, "AG/TC": -22.2, "GA/CT": -22.2,
        "CG/GC": -27.2, "GC/CG": -24.4, "GG/CC": -19.9, "CC/GG": -19.9
    }


class DNASequence:
    """
    Class representing a DNA sequence with validation and basic operations.

    Attributes:
        sequence (str): The DNA sequence in uppercase
    """

    def __init__(self, sequence: str):
        """Initialize DNA sequence with validation."""
        self.sequence = self._validate_sequence(sequence.upper())

    @staticmethod
    def _validate_sequence(sequence: str) -> str:
        """Validate DNA sequence contains only valid bases."""
        if not set(sequence).issubset(DNA_BASES):
            invalid_bases = set(sequence) - DNA_BASES
            raise ValueError(f"Invalid DNA bases found: {invalid_bases}")
        return sequence

    def complement(self) -> 'DNASequence':
        """Return the complement sequence."""
        complement = ''.join(COMPLEMENTARY_BASES[base] for base in self.sequence)
        return DNASequence(complement)

    def reverse_complement(self) -> 'DNASequence':
        """Return the reverse complement sequence."""
        return DNASequence(self.complement().sequence[::-1])

    def gc_content(self) -> float:
        """Calculate GC content as a percentage."""
        gc_count = sum(1 for base in self.sequence if base in {'G', 'C'})
        return (gc_count / len(self.sequence)) * 100

    def __len__(self) -> int:
        return len(self.sequence)

    def __str__(self) -> str:
        return self.sequence


class PatternAnalyzer:
    """Class for analyzing patterns within DNA sequences."""

    @staticmethod
    def find_pattern(sequence: DNASequence, pattern: str) -> List[int]:
        """
        Find all occurrences of a pattern in the sequence.

        Args:
            sequence: DNASequence object to search in
            pattern: Pattern to search for

        Returns:
            List of starting positions where pattern was found
        """
        pattern = pattern.upper()
        locations = []

        for i in range(len(sequence) - len(pattern) + 1):
            if sequence.sequence[i:i + len(pattern)] == pattern:
                locations.append(i)

        return locations

    @staticmethod
    def find_repeats(sequence: DNASequence, length: int) -> Dict[str, int]:
        """
        Find all repeating sequences of specified length.

        Args:
            sequence: DNASequence object to analyze
            length: Length of repeat sequences to look for

        Returns:
            Dictionary of sequences and their counts
        """
        counts = {}

        for i in range(len(sequence) - length + 1):
            substr = sequence.sequence[i:i + length]
            counts[substr] = counts.get(substr, 0) + 1

        return {seq: count for seq, count in counts.items() if count > 1}


class SequenceGenerator:
    """Class for generating DNA sequences."""

    @staticmethod
    def random_sequence(length: int) -> DNASequence:
        """Generate random DNA sequence of specified length."""
        sequence = ''.join(random.choice(list(DNA_BASES)) for _ in range(length))
        return DNASequence(sequence)

    @staticmethod
    def gc_specific_sequence(length: int, gc_count: int) -> DNASequence:
        """
        Generate DNA sequence with specific GC content.

        Args:
            length: Desired sequence length
            gc_count: Number of G/C bases desired
        """
        if gc_count > length:
            raise ValueError("GC count cannot exceed sequence length")

        bases = ['G', 'C'] * gc_count + ['A', 'T'] * (length - gc_count)
        random.shuffle(bases)
        return DNASequence(''.join(bases))


class SequenceAligner:
    """Implementation of sequence alignment algorithms."""

    @staticmethod
    def smith_waterman(seq1: DNASequence, seq2: DNASequence,
                       match: int = 2, mismatch: int = -1, gap: int = -1) -> Tuple[str, str, float]:
        """
        Implement Smith-Waterman algorithm for local sequence alignment.

        Args:
            seq1, seq2: Sequences to align
            match: Score for matching bases
            mismatch: Penalty for mismatched bases
            gap: Penalty for gaps

        Returns:
            Tuple of (alignment1, alignment2, score)

        Reference:
            Smith TF, Waterman MS. (1981)
        """
        m, n = len(seq1), len(seq2)
        score_matrix = pd.DataFrame(0, index=range(m + 1), columns=range(n + 1))

        # Fill score matrix
        max_score = 0
        max_pos = (0, 0)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                score = match if seq1.sequence[i - 1] == seq2.sequence[j - 1] else mismatch
                score_matrix.at[i, j] = max(0,
                                            score_matrix.at[i - 1, j - 1] + score,  # diagonal
                                            score_matrix.at[i - 1, j] + gap,  # vertical
                                            score_matrix.at[i, j - 1] + gap  # horizontal
                                            )

                if score_matrix.at[i, j] > max_score:
                    max_score = score_matrix.at[i, j]
                    max_pos = (i, j)

        # Traceback
        align1, align2 = '', ''
        i, j = max_pos

        while score_matrix.at[i, j] != 0:
            score = match if seq1.sequence[i - 1] == seq2.sequence[j - 1] else mismatch
            if score_matrix.at[i, j] == score_matrix.at[i - 1, j - 1] + score:
                align1 = seq1.sequence[i - 1] + align1
                align2 = seq2.sequence[j - 1] + align2
                i -= 1
                j -= 1
            elif score_matrix.at[i, j] == score_matrix.at[i - 1, j] + gap:
                align1 = seq1.sequence[i - 1] + align1
                align2 = '-' + align2
                i -= 1
            else:
                align1 = '-' + align1
                align2 = seq2.sequence[j - 1] + align2
                j -= 1

        return align1, align2, max_score


class ThermodynamicCalculator:
    """Class for calculating thermodynamic properties of DNA sequences."""

    def __init__(self):
        self.params = ThermodynamicParameters()

    def calculate_tm(self, sequence: DNASequence, complement: DNASequence,
                     concentration: float) -> float:
        """
        Calculate melting temperature using nearest-neighbor method.

        Args:
            sequence: Forward sequence
            complement: Complement sequence
            concentration: Molar concentration of oligonucleotide

        Returns:
            Melting temperature in Celsius

        Reference:
            SantaLucia J Jr. (1998)
        """
        if len(sequence) != len(complement):
            raise ValueError("Sequences must be same length")

        # Initialize with terminal base pairs
        delta_h = 0.1 if sequence.sequence[0] in 'GC' else 2.3
        delta_s = -2.8 if sequence.sequence[0] in 'GC' else 4.1

        # Add correction for complementarity
        is_complementary = str(sequence.complement()) == str(complement)
        delta_h -= 1.4 if is_complementary else 0
        concentration_divisor = 1 if is_complementary else 4

        # Calculate nearest-neighbor contributions
        for i in range(len(sequence) - 1):
            pair = f"{sequence.sequence[i:i + 2]}/{complement.sequence[i:i + 2]}"
            delta_h += self.params.DELTA_H.get(pair, 0)
            delta_s += self.params.DELTA_S.get(pair, 0)

        # Calculate Tm
        tm = (delta_h * 1000 /
              (delta_s + R_CONSTANT * np.log(concentration / concentration_divisor)) - 273.15)

        return round(tm, 2)


class DNAAnalyzer:
    """
    Main interface for DNA analysis operations.

    This class provides a unified interface to access all DNA analysis
    functionality through a single point of entry.
    """

    def __init__(self):
        self.pattern_analyzer = PatternAnalyzer()
        self.sequence_generator = SequenceGenerator()
        self.sequence_aligner = SequenceAligner()
        self.thermo_calculator = ThermodynamicCalculator()

    def analyze_sequence(self, sequence: str) -> Dict:
        """
        Perform comprehensive analysis of a DNA sequence.

        Returns dictionary with various sequence properties and analyses.
        """
        dna = DNASequence(sequence)

        return {
            'sequence': str(dna),
            'length': len(dna),
            'gc_content': dna.gc_content(),
            'complement': str(dna.complement()),
            'reverse_complement': str(dna.reverse_complement()),
            'repeats': self.pattern_analyzer.find_repeats(dna, 4)  # Default 4-mer repeats
        }

    def generate_random_sequence(self, length: int, gc_content: Optional[float] = None) -> str:
        """Generate random sequence with optional GC content specification."""
        if gc_content is not None:
            gc_count = int(length * gc_content / 100)
            return str(self.sequence_generator.gc_specific_sequence(length, gc_count))
        return str(self.sequence_generator.random_sequence(length))

    def align_sequences(self, seq1: str, seq2: str) -> Dict:
        """Align two sequences and return alignment details."""
        dna1 = DNASequence(seq1)
        dna2 = DNASequence(seq2)
        align1, align2, score = self.sequence_aligner.smith_waterman(dna1, dna2)

        return {
            'alignment1': align1,
            'alignment2': align2,
            'score': score,
            'identity': sum(a == b for a, b in zip(align1, align2)) / len(align1) * 100
        }

    def calculate_melting_temp(self, sequence: str, concentration: float) -> float:
        """Calculate melting temperature for a sequence."""
        dna = DNASequence(sequence)
        complement = dna.complement()
        return self.thermo_calculator.calculate_tm(dna, complement, concentration)


def main():
    """Example usage of the DNA Analysis Tool."""
    analyzer = DNAAnalyzer()

    # Example sequence
    sequence = "ATCGATCGATCG"

    # Basic analysis
    print("Basic sequence analysis:")
    analysis = analyzer.analyze_sequence(sequence)
    for key, value in analysis.items():
        print(f"Alignment 1: {alignment['alignment1']}")
    print(f"Alignment 2: {alignment['alignment2']}")

    # Melting temperature calculation
    print("\nMelting temperature calculation:")
    tm = analyzer.calculate_melting_temp(sequence, concentration=1e-6)
    print(f"Melting temperature: {tm}°C")


class CLI:
    """Command Line Interface for DNA Analysis Tool."""

    def __init__(self):
        self.analyzer = DNAAnalyzer()
        self.menu_options = {
            1: ("Analyze DNA sequence", self.analyze_sequence),
            2: ("Generate random sequence", self.generate_sequence),
            3: ("Align sequences", self.align_sequences),
            4: ("Calculate melting temperature", self.calculate_tm),
            5: ("Exit", self.exit_program)
        }
        self.running = True

    def display_menu(self):
        """Display main menu options."""
        print("\nDNA Analysis Tool")
        print("-----------------")
        for key, (option, _) in self.menu_options.items():
            print(f"{key}. {option}")

    def get_valid_input(self, prompt: str, validator=None) -> str:
        """Get validated input from user."""
        while True:
            user_input = input(prompt).strip()
            if validator is None or validator(user_input):
                return user_input
            print("Invalid input. Please try again.")

    def analyze_sequence(self):
        """Handle sequence analysis option."""
        sequence = self.get_valid_input("Enter DNA sequence: ",
                                        lambda s: set(s.upper()).issubset(DNA_BASES))
        analysis = self.analyzer.analyze_sequence(sequence)

        print("\nAnalysis Results:")
        print("----------------")
        for key, value in analysis.items():
            print(f"{key}: {value}")

    def generate_sequence(self):
        """Handle sequence generation option."""
        length = int(self.get_valid_input("Enter desired sequence length: ",
                                          lambda s: s.isdigit() and int(s) > 0))

        gc_prompt = self.get_valid_input("Specify GC content? (y/n): ",
                                         lambda s: s.lower() in ['y', 'n'])

        if gc_prompt.lower() == 'y':
            gc_content = float(self.get_valid_input("Enter GC content percentage (0-100): ",
                                                    lambda s: s.replace('.', '').isdigit() and 0 <= float(s) <= 100))
            sequence = self.analyzer.generate_random_sequence(length, gc_content)
        else:
            sequence = self.analyzer.generate_random_sequence(length)

        print(f"\nGenerated sequence: {sequence}")

    def align_sequences(self):
        """Handle sequence alignment option."""
        seq1 = self.get_valid_input("Enter first sequence: ",
                                    lambda s: set(s.upper()).issubset(DNA_BASES))
        seq2 = self.get_valid_input("Enter second sequence: ",
                                    lambda s: set(s.upper()).issubset(DNA_BASES))

        alignment = self.analyzer.align_sequences(seq1, seq2)

        print("\nAlignment Results:")
        print("-----------------")
        print(f"Sequence 1: {alignment['alignment1']}")
        print(f"Sequence 2: {alignment['alignment2']}")
        print(f"Alignment score: {alignment['score']}")
        print(f"Sequence identity: {alignment['identity']}%")

    def calculate_tm(self):
        """Handle melting temperature calculation option."""
        sequence = self.get_valid_input("Enter DNA sequence: ",
                                        lambda s: set(s.upper()).issubset(DNA_BASES))

        concentration = float(self.get_valid_input(
            "Enter oligonucleotide concentration (in M, e.g., 1e-6): ",
            lambda s: s.replace('e', '').replace('-', '').replace('.', '').isdigit()
        ))

        tm = self.analyzer.calculate_melting_temp(sequence, concentration)
        print(f"\nMelting temperature: {tm}°C")

    def exit_program(self):
        """Handle program exit."""
        print("\nThank you for using DNA Analysis Tool!")
        self.running = False

    def run(self):
        """Main program loop."""
        print("\nWelcome to DNA Analysis Tool!")

        while self.running:
            self.display_menu()
            try:
                choice = int(self.get_valid_input("\nEnter your choice (1-5): ",
                                                  lambda s: s.isdigit() and 1 <= int(s) <= 5))
                self.menu_options[choice][1]()
            except ValueError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    """
    DNA Analysis Tool
    ----------------
    A comprehensive toolkit for DNA sequence analysis and manipulation.

    This module provides a collection of tools for:
    - DNA sequence validation and manipulation
    - Sequence property analysis (GC content, melting temperature)
    - Pattern searching and alignment
    - Random sequence generation
    - Thermodynamic calculations

    References:
    -----------
    [1] SantaLucia J Jr. (1998) "A unified view of polymer, dumbbell, and oligonucleotide DNA 
        nearest-neighbor thermodynamics", PNAS, 95(4):1460-1465.
    [2] Smith TF, Waterman MS. (1981) "Identification of common molecular subsequences", 
        J Mol Biol, 147(1):195-197.
    [3] Breslauer KJ et al. (1986) "Predicting DNA duplex stability from the base sequence", 
        PNAS, 83(11):3746-3750.
    """

    import random
    from dataclasses import dataclass
    from typing import Dict, List, Tuple, Optional, Union
    import numpy as np
    import pandas as pd
    from abc import ABC, abstractmethod

    # Constants
    DNA_BASES = {'A', 'T', 'C', 'G'}
    COMPLEMENTARY_BASES = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    R_CONSTANT = 1.987  # cal/(K*mol) - Gas constant


    @dataclass
    class ThermodynamicParameters:
        """Storage class for nearest-neighbor thermodynamic parameters."""

        # ΔH° values (kcal/mol) for nearest-neighbor pairs
        # Data from SantaLucia (1998)
        DELTA_H = {
            "AA/TT": -7.9, "TT/AA": -7.9, "AT/TA": -7.2, "TA/AT": -7.2,
            "CA/GT": -8.5, "AC/TG": -8.5, "GT/CA": -8.4, "TG/AC": -8.4,
            "CT/GA": -7.8, "TC/AG": -7.8, "AG/TC": -8.2, "GA/CT": -8.2,
            "CG/GC": -10.6, "GC/CG": -9.8, "GG/CC": -8.0, "CC/GG": -8.0
        }

        # ΔS° values (cal/K·mol) for nearest-neighbor pairs
        DELTA_S = {
            "AA/TT": -22.2, "TT/AA": -22.2, "AT/TA": -20.4, "TA/AT": -21.3,
            "CA/GT": -22.7, "AC/TG": -22.7, "GT/CA": -22.4, "TG/AC": -8.4,
            "CT/GA": -21.0, "TC/AG": -21.0, "AG/TC": -22.2, "GA/CT": -22.2,
            "CG/GC": -27.2, "GC/CG": -24.4, "GG/CC": -19.9, "CC/GG": -19.9
        }


    class DNASequence:
        """
        Class representing a DNA sequence with validation and basic operations.

        Attributes:
            sequence (str): The DNA sequence in uppercase
        """

        def __init__(self, sequence: str):
            """Initialize DNA sequence with validation."""
            self.sequence = self._validate_sequence(sequence.upper())

        @staticmethod
        def _validate_sequence(sequence: str) -> str:
            """Validate DNA sequence contains only valid bases."""
            if not set(sequence).issubset(DNA_BASES):
                invalid_bases = set(sequence) - DNA_BASES
                raise ValueError(f"Invalid DNA bases found: {invalid_bases}")
            return sequence

        def complement(self) -> 'DNASequence':
            """Return the complement sequence."""
            complement = ''.join(COMPLEMENTARY_BASES[base] for base in self.sequence)
            return DNASequence(complement)

        def reverse_complement(self) -> 'DNASequence':
            """Return the reverse complement sequence."""
            return DNASequence(self.complement().sequence[::-1])

        def gc_content(self) -> float:
            """Calculate GC content as a percentage."""
            gc_count = sum(1 for base in self.sequence if base in {'G', 'C'})
            return (gc_count / len(self.sequence)) * 100

        def __len__(self) -> int:
            return len(self.sequence)

        def __str__(self) -> str:
            return self.sequence


    class PatternAnalyzer:
        """Class for analyzing patterns within DNA sequences."""

        @staticmethod
        def find_pattern(sequence: DNASequence, pattern: str) -> List[int]:
            """
            Find all occurrences of a pattern in the sequence.

            Args:
                sequence: DNASequence object to search in
                pattern: Pattern to search for

            Returns:
                List of starting positions where pattern was found
            """
            pattern = pattern.upper()
            locations = []

            for i in range(len(sequence) - len(pattern) + 1):
                if sequence.sequence[i:i + len(pattern)] == pattern:
                    locations.append(i)

            return locations

        @staticmethod
        def find_repeats(sequence: DNASequence, length: int) -> Dict[str, int]:
            """
            Find all repeating sequences of specified length.

            Args:
                sequence: DNASequence object to analyze
                length: Length of repeat sequences to look for

            Returns:
                Dictionary of sequences and their counts
            """
            counts = {}

            for i in range(len(sequence) - length + 1):
                substr = sequence.sequence[i:i + length]
                counts[substr] = counts.get(substr, 0) + 1

            return {seq: count for seq, count in counts.items() if count > 1}


    class SequenceGenerator:
        """Class for generating DNA sequences."""

        @staticmethod
        def random_sequence(length: int) -> DNASequence:
            """Generate random DNA sequence of specified length."""
            sequence = ''.join(random.choice(list(DNA_BASES)) for _ in range(length))
            return DNASequence(sequence)

        @staticmethod
        def gc_specific_sequence(length: int, gc_count: int) -> DNASequence:
            """
            Generate DNA sequence with specific GC content.

            Args:
                length: Desired sequence length
                gc_count: Number of G/C bases desired
            """
            if gc_count > length:
                raise ValueError("GC count cannot exceed sequence length")

            bases = ['G', 'C'] * gc_count + ['A', 'T'] * (length - gc_count)
            random.shuffle(bases)
            return DNASequence(''.join(bases))


    class SequenceAligner:
        """Implementation of sequence alignment algorithms."""

        @staticmethod
        def smith_waterman(seq1: DNASequence, seq2: DNASequence,
                           match: int = 2, mismatch: int = -1, gap: int = -1) -> Tuple[str, str, float]:
            """
            Implement Smith-Waterman algorithm for local sequence alignment.

            Args:
                seq1, seq2: Sequences to align
                match: Score for matching bases
                mismatch: Penalty for mismatched bases
                gap: Penalty for gaps

            Returns:
                Tuple of (alignment1, alignment2, score)

            Reference:
                Smith TF, Waterman MS. (1981)
            """
            m, n = len(seq1), len(seq2)
            score_matrix = pd.DataFrame(0, index=range(m + 1), columns=range(n + 1))

            # Fill score matrix
            max_score = 0
            max_pos = (0, 0)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    score = match if seq1.sequence[i - 1] == seq2.sequence[j - 1] else mismatch
                    score_matrix.at[i, j] = max(0,
                                                score_matrix.at[i - 1, j - 1] + score,  # diagonal
                                                score_matrix.at[i - 1, j] + gap,  # vertical
                                                score_matrix.at[i, j - 1] + gap  # horizontal
                                                )

                    if score_matrix.at[i, j] > max_score:
                        max_score = score_matrix.at[i, j]
                        max_pos = (i, j)

            # Traceback
            align1, align2 = '', ''
            i, j = max_pos

            while score_matrix.at[i, j] != 0:
                score = match if seq1.sequence[i - 1] == seq2.sequence[j - 1] else mismatch
                if score_matrix.at[i, j] == score_matrix.at[i - 1, j - 1] + score:
                    align1 = seq1.sequence[i - 1] + align1
                    align2 = seq2.sequence[j - 1] + align2
                    i -= 1
                    j -= 1
                elif score_matrix.at[i, j] == score_matrix.at[i - 1, j] + gap:
                    align1 = seq1.sequence[i - 1] + align1
                    align2 = '-' + align2
                    i -= 1
                else:
                    align1 = '-' + align1
                    align2 = seq2.sequence[j - 1] + align2
                    j -= 1

            return align1, align2, max_score


    class ThermodynamicCalculator:
        """Class for calculating thermodynamic properties of DNA sequences."""

        def __init__(self):
            self.params = ThermodynamicParameters()

        def calculate_tm(self, sequence: DNASequence, complement: DNASequence,
                         concentration: float) -> float:
            """
            Calculate melting temperature using nearest-neighbor method.

            Args:
                sequence: Forward sequence
                complement: Complement sequence
                concentration: Molar concentration of oligonucleotide

            Returns:
                Melting temperature in Celsius

            Reference:
                SantaLucia J Jr. (1998)
            """
            if len(sequence) != len(complement):
                raise ValueError("Sequences must be same length")

            # Initialize with terminal base pairs
            delta_h = 0.1 if sequence.sequence[0] in 'GC' else 2.3
            delta_s = -2.8 if sequence.sequence[0] in 'GC' else 4.1

            # Add correction for complementarity
            is_complementary = str(sequence.complement()) == str(complement)
            delta_h -= 1.4 if is_complementary else 0
            concentration_divisor = 1 if is_complementary else 4

            # Calculate nearest-neighbor contributions
            for i in range(len(sequence) - 1):
                pair = f"{sequence.sequence[i:i + 2]}/{complement.sequence[i:i + 2]}"
                delta_h += self.params.DELTA_H.get(pair, 0)
                delta_s += self.params.DELTA_S.get(pair, 0)

            # Calculate Tm
            tm = (delta_h * 1000 /
                  (delta_s + R_CONSTANT * np.log(concentration / concentration_divisor)) - 273.15)

            return round(tm, 2)


    class DNAAnalyzer:
        """
        Main interface for DNA analysis operations.

        This class provides a unified interface to access all DNA analysis
        functionality through a single point of entry.
        """

        def __init__(self):
            self.pattern_analyzer = PatternAnalyzer()
            self.sequence_generator = SequenceGenerator()
            self.sequence_aligner = SequenceAligner()
            self.thermo_calculator = ThermodynamicCalculator()

        def analyze_sequence(self, sequence: str) -> Dict:
            """
            Perform comprehensive analysis of a DNA sequence.

            Returns dictionary with various sequence properties and analyses.
            """
            dna = DNASequence(sequence)

            return {
                'sequence': str(dna),
                'length': len(dna),
                'gc_content': dna.gc_content(),
                'complement': str(dna.complement()),
                'reverse_complement': str(dna.reverse_complement()),
                'repeats': self.pattern_analyzer.find_repeats(dna, 4)  # Default 4-mer repeats
            }

        def generate_random_sequence(self, length: int, gc_content: Optional[float] = None) -> str:
            """Generate random sequence with optional GC content specification."""
            if gc_content is not None:
                gc_count = int(length * gc_content / 100)
                return str(self.sequence_generator.gc_specific_sequence(length, gc_count))
            return str(self.sequence_generator.random_sequence(length))

        def align_sequences(self, seq1: str, seq2: str) -> Dict:
            """Align two sequences and return alignment details."""
            dna1 = DNASequence(seq1)
            dna2 = DNASequence(seq2)
            align1, align2, score = self.sequence_aligner.smith_waterman(dna1, dna2)

            return {
                'alignment1': align1,
                'alignment2': align2,
                'score': score,
                'identity': sum(a == b for a, b in zip(align1, align2)) / len(align1) * 100
            }

        def calculate_melting_temp(self, sequence: str, concentration: float) -> float:
            """Calculate melting temperature for a sequence."""
            dna = DNASequence(sequence)
            complement = dna.complement()
            return self.thermo_calculator.calculate_tm(dna, complement, concentration)


    def main():
        """Example usage of the DNA Analysis Tool."""
        analyzer = DNAAnalyzer()

        # Example sequence
        sequence = "ATCGATCGATCG"

        # Basic analysis
        print("Basic sequence analysis:")
        analysis = analyzer.analyze_sequence(sequence)
        for key, value in analysis.items():
            print(f"Alignment 1: {alignment['alignment1']}")
        print(f"Alignment 2: {alignment['alignment2']}")

        # Melting temperature calculation
        print("\nMelting temperature calculation:")
        tm = analyzer.calculate_melting_temp(sequence, concentration=1e-6)
        print(f"Melting temperature: {tm}°C")


    class CLI:
        """Command Line Interface for DNA Analysis Tool."""

        def __init__(self):
            self.analyzer = DNAAnalyzer()
            self.menu_options = {
                1: ("Analyze DNA sequence", self.analyze_sequence),
                2: ("Generate random sequence", self.generate_sequence),
                3: ("Align sequences", self.align_sequences),
                4: ("Calculate melting temperature", self.calculate_tm),
                5: ("Exit", self.exit_program)
            }
            self.running = True

        def display_menu(self):
            """Display main menu options."""
            print("\nDNA Analysis Tool")
            print("-----------------")
            for key, (option, _) in self.menu_options.items():
                print(f"{key}. {option}")

        def get_valid_input(self, prompt: str, validator=None) -> str:
            """Get validated input from user."""
            while True:
                user_input = input(prompt).strip()
                if validator is None or validator(user_input):
                    return user_input
                print("Invalid input. Please try again.")

        def analyze_sequence(self):
            """Handle sequence analysis option."""
            sequence = self.get_valid_input("Enter DNA sequence: ",
                                            lambda s: set(s.upper()).issubset(DNA_BASES))
            analysis = self.analyzer.analyze_sequence(sequence)

            print("\nAnalysis Results:")
            print("----------------")
            for key, value in analysis.items():
                print(f"{key}: {value}")

        def generate_sequence(self):
            """Handle sequence generation option."""
            length = int(self.get_valid_input("Enter desired sequence length: ",
                                              lambda s: s.isdigit() and int(s) > 0))

            gc_prompt = self.get_valid_input("Specify GC content? (y/n): ",
                                             lambda s: s.lower() in ['y', 'n'])

            if gc_prompt.lower() == 'y':
                gc_content = float(self.get_valid_input("Enter GC content percentage (0-100): ",
                                                        lambda s: s.replace('.', '').isdigit() and 0 <= float(
                                                            s) <= 100))
                sequence = self.analyzer.generate_random_sequence(length, gc_content)
            else:
                sequence = self.analyzer.generate_random_sequence(length)

            print(f"\nGenerated sequence: {sequence}")

        def align_sequences(self):
            """Handle sequence alignment option."""
            seq1 = self.get_valid_input("Enter first sequence: ",
                                        lambda s: set(s.upper()).issubset(DNA_BASES))
            seq2 = self.get_valid_input("Enter second sequence: ",
                                        lambda s: set(s.upper()).issubset(DNA_BASES))

            alignment = self.analyzer.align_sequences(seq1, seq2)

            print("\nAlignment Results:")
            print("-----------------")
            print(f"Sequence 1: {alignment['alignment1']}")
            print(f"Sequence 2: {alignment['alignment2']}")
            print(f"Alignment score: {alignment['score']}")
            print(f"Sequence identity: {alignment['identity']}%")

        def calculate_tm(self):
            """Handle melting temperature calculation option."""
            sequence = self.get_valid_input("Enter DNA sequence: ",
                                            lambda s: set(s.upper()).issubset(DNA_BASES))

            concentration = float(self.get_valid_input(
                "Enter oligonucleotide concentration (in M, e.g., 1e-6): ",
                lambda s: s.replace('e', '').replace('-', '').replace('.', '').isdigit()
            ))

            tm = self.analyzer.calculate_melting_temp(sequence, concentration)
            print(f"\nMelting temperature: {tm}°C")

        def exit_program(self):
            """Handle program exit."""
            print("\nThank you for using DNA Analysis Tool!")
            self.running = False

        def run(self):
            """Main program loop."""
            print("\nWelcome to DNA Analysis Tool!")

            while self.running:
                self.display_menu()
                try:
                    choice = int(self.get_valid_input("\nEnter your choice (1-5): ",
                                                      lambda s: s.isdigit() and 1 <= int(s) <= 5))
                    self.menu_options[choice][1]()
                except ValueError as e:
                    print(f"Error: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="DNA Analysis Tool")
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run example usage demonstrating key features"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive command-line interface"
    )

    args = parser.parse_args()

    # If no arguments provided, show help
    if not (args.example or args.interactive):
        parser.print_help()
        exit(1)

    # Run examples if requested
    if args.example:
        print("Running example usage demonstrations...")
        main()

    # Start interactive CLI if requested
    if args.interactive:
        print("\nStarting interactive interface...")
        cli = CLI()
        cli.run()
