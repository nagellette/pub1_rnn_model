'''
Math functions to created tailered versions as needed in the codes.
'''


class math_support_functions:
    '''
    Function returning always the below floor.
    Usually needed for ranging operations to avoid array indexing errors
    '''

    def round_to_floor(number, divider):
        occurance = round(number / divider)
        if number - (occurance * divider) < 0:
            return occurance - 1
        else:
            return occurance


'''
Landsat-5 (TM) based standart data manupilation functions
'''


class landsat_support_functions:
    '''
    Normalize np data typed pixel values to 0.0-1.0 range.
        * Saturate values (20,000) converted to highest reflectance 16000
        * No date values (-9999) converted to -1
    '''

    def normalize_band(band):
        # replace saturate values with highest reflectance
        band[band == 20000] = 16000

        # normalize pixel values with shif from -2000 to 0, original data range is -2000-16000
        band = (band + 2000.0) / 18000.0

        # replace no data values to -1 from -9999
        band[band == band.min()] = -1
        return band
