import pickle


def main():
    subroutines = pickle.load(open("Fortran_subroutines.pkl", "rb"))
    petpt_code = subroutines[("dssat-csm/SPAM", "PET.for", "PETASCE")]
    print("\n".join(petpt_code))


if __name__ == '__main__':
    main()
