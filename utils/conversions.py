def pixel_to_distance(pixel_distance, height_in_meters, height_in_pixels):
    return (pixel_distance * height_in_meters) / height_in_pixels


def distance_to_pixel(distance, height_in_meters, height_in_pixels):
    return (distance * height_in_pixels) / height_in_meters
