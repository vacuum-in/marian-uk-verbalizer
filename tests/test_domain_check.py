from src.verbalizer.evaluation.domain_check import coords_ok, digits_ok, english_ok, iban_ok, phone_ok, score_row


def test_english_variants() -> None:
    assert english_ok("пайтон", ["пайтон"])
    assert english_ok("дата", ["дейта"])
    assert english_ok("директор", ["дайректор"])
    assert english_ok("хоум", ["гоум"])
    assert english_ok("кідз", ["кідс"])
    assert not english_ok("еврістайм", ["еврітайм", "еверітайм"])
    assert not english_ok("блок", ["бленк"])


def test_digit_strings_any_grouping() -> None:
    assert digits_ok("п'ять три сім п'ять", "5375")
    assert digits_ok("п'ятдесят три сімдесят п'ять", "5375")
    assert digits_ok("нуль сорок чотири триста дев'яносто дванадцять тридцять чотири", "0443901234")
    assert digits_ok("чотириста тридцять мільйонів сімсот сім тисяч", "430707000")
    assert not digits_ok("чотири нуль нуль", "4000")
    assert not digits_ok("два три", "5")
    assert not digits_ok("п'ять три сім п'ять гривень", "5375")


def test_iban_and_phone() -> None:
    assert iban_ok("ю ей нуль один, чотири нуль нуль нуль", "UA", "014000")
    assert not iban_ok("ю ей нуль один, чотири нуль нуль", "UA", "014000")
    assert phone_ok("плюс три вісім нуль сорок чотири двісті десять двадцять", "380442001020", True)
    assert not phone_ok("три вісім нуль сорок чотири двісті десять двадцять", "380442001020", True)


def test_coordinates() -> None:
    reading = "сорок вісім градусів п'ятдесят п'ять хвилин дванадцять секунд північної широти"
    assert coords_ok(reading, [48, 55, 12], ["deg", "min", "sec"], ["N"])
    assert not coords_ok(reading.replace("північної", "південної"), [48, 55, 12], ["deg", "min", "sec"], ["N"])
    assert not coords_ok("сорок вісім градусів п'ятдесят п'ять хвилин дванадцять хвилин північної широти",
                         [48, 55, 12], ["deg", "min", "sec"], ["N"])
    assert coords_ok("сімдесят три градуси тридцять три хвилини північної широти", [73, 33, 0], ["deg", "min", "sec"], ["N"])
    assert coords_ok("п'ятдесят цілих чотири тисячі п'ятсот одна десятитисячна градуса північної широти",
                     [50, 4501], ["deg"], ["N"])


def test_frame_alignment_and_damage() -> None:
    row = {"frame": ["У відділі великий ", " ", "."], "slots": [
        {"kind": "english", "variants": ["біг"]}, {"kind": "english", "variants": ["дейта"]}]}
    assert score_row(row, "У відділі великий біґ дата.")["ok"]
    damaged = score_row(row, "У відділі велик біг дата.")
    assert not damaged["ok"] and damaged["reason"] == "fixed text changed"


def test_coordinates_reject_the_russianism_minuta() -> None:
    assert not coords_ok("сорок вісім градусів п'ятдесят п'ять мінут північної широти", [48, 55], ["deg", "min"], ["N"])
    assert coords_ok("сорок вісім градусів п'ятдесят п'ять хвилин північної широти", [48, 55], ["deg", "min"], ["N"])
