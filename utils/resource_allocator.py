def allocate_resources(predicted_cases):

    if predicted_cases < 50:
        status = "Normal Monitoring"
        beds = predicted_cases * 0.1
        doctors = predicted_cases * 0.03

    elif predicted_cases < 150:
        status = "Increase Doctors"
        beds = predicted_cases * 0.2
        doctors = predicted_cases * 0.05

    elif predicted_cases < 300:
        status = "Allocate Hospital Beds"
        beds = predicted_cases * 0.3
        doctors = predicted_cases * 0.08

    else:
        status = "Emergency Response Required"
        beds = predicted_cases * 0.5
        doctors = predicted_cases * 0.1

    return status, int(beds), int(doctors)